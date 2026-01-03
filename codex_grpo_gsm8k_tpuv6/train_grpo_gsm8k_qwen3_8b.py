from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import easydel as ed
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer


def _extract_gsm8k_label(answer_text: str) -> int:
    match = re.search(r"####\\s*([-+]?\\d+)", answer_text)
    if match:
        return int(match.group(1))
    nums = re.findall(r"[-+]?\\d+", answer_text)
    if nums:
        return int(nums[-1])
    raise ValueError(f"Could not parse GSM8K label from: {answer_text!r}")


def _extract_final_int(text: str) -> int | None:
    match = re.search(r"Final Answer\\s*[:：]\\s*([-+]?\\d[\\d,]*)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))
    match = re.search(r"####\\s*([-+]?\\d+)", text)
    if match:
        return int(match.group(1))
    nums = re.findall(r"[-+]?\\d+", text)
    if nums:
        return int(nums[-1])
    return None


def gsm8k_exact_match_reward(prompts: list[Any], completions: list[Any], batch: dict[str, Any], **_: Any) -> list[float]:
    labels = batch.get("label")
    if labels is None:
        return [0.0] * len(completions)

    try:
        labels_list = [int(x) for x in list(labels)]
    except Exception:
        labels_list = [int(labels)]

    denom = max(len(labels_list), 1)
    generation_factor = max(len(completions) // denom, 1)

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            completion_text = str(completion[0].get("content", ""))
        else:
            completion_text = str(completion)

        pred = _extract_final_int(completion_text)
        gold = labels_list[min(i // generation_factor, denom - 1)]
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


def _build_gsm8k_dataset(train_examples: int):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if train_examples is not None and train_examples > 0:
        dataset = dataset.select(range(min(train_examples, len(dataset))))

    system_prompt = (
        "Solve the math word problem. Think step-by-step, then output the final answer as:\n"
        "Final Answer: <integer>\n"
    )

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        label = _extract_gsm8k_label(example["answer"])
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
        return {"prompt": prompt, "label": label}

    return dataset.map(_map, remove_columns=list(dataset.column_names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--train_examples", type=int, default=256)

    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=256)

    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--total_batch_size", type=int, default=0, help="0 => jax.device_count()")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.04)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_target_modules", type=str, default=".*q_proj.*")

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="")

    args = parser.parse_args()

    total_batch_size = args.total_batch_size if args.total_batch_size > 0 else max(jax.device_count(), 1)
    run_dir = Path(args.save_dir) if args.save_dir else (Path(__file__).resolve().parent / "outputs" / "gsm8k_grpo_10steps")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    except ValueError as exc:
        if "tiktoken" in str(exc).lower():
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
        else:
            raise
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_sequence_length = args.max_prompt_length + args.max_completion_length
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        args.model_id,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        backend="tpu",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            attn_mechanism=ed.AttentionMechanisms.SDPA,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    if args.lora_rank > 0:
        model = model.apply_lora_to_layers(rank=args.lora_rank, target_modules=args.lora_target_modules)

    train_dataset = _build_gsm8k_dataset(args.train_examples)

    config = ed.GRPOConfig(
        model_name="qwen3_8b_gsm8k_grpo_smoketest",
        save_directory=str(run_dir),
        backend="tpu",
        max_training_steps=args.max_steps,
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_return_sequences=args.num_return_sequences,
        learning_rate=args.learning_rate,
        beta=args.beta,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.NONE,
        clip_grad=1.0,
        use_wandb=args.use_wandb,
        do_last_save=False,
    )

    trainer = ed.GRPOTrainer(
        arguments=config,
        model=model,
        reward_funcs=gsm8k_exact_match_reward,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
