# GSM8K + GRPO (10 steps) on TPU v6 (EasyDeL)

Minimal smoke-test to confirm EasyDeL `GRPOTrainer` runs end-to-end on a TPU v6 VM with a Qwen 8B-class model.

## Quickstart (TPU VM)

```bash
git clone https://github.com/demon2036/easydel-gsm8k-grpo-tpuv6.git
cd easydel-gsm8k-grpo-tpuv6

bash codex_grpo_gsm8k_tpuv6/run_tpu.sh
```

## What it does

- Loads `openai/gsm8k` (`main` split) and builds chat-style prompts.
- Uses `easydel.GRPOTrainer` to run `max_training_steps=10`.
- Reward: exact-match on the final integer answer (`1.0` correct, `0.0` otherwise).

## Customize

Run directly:

```bash
source codex_grpo_gsm8k_tpuv6/.venv/bin/activate
python codex_grpo_gsm8k_tpuv6/train_grpo_gsm8k_qwen3_8b.py \
  --model_id Qwen/Qwen3-8B-Instruct \
  --max_steps 10
```

Useful flags:

- `--model_id`: HF model id (default: `Qwen/Qwen3-8B-Instruct`)
- `--max_steps`: training steps (default: `10`)
- `--train_examples`: number of GSM8K training rows to use (default: `256`)
- `--num_return_sequences`: completions per prompt (default: `4`)
- `--total_batch_size`: global batch size (default: `jax.device_count()`)
