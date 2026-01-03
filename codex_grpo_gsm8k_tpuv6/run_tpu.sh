#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

cd "${ROOT_DIR}"

python3 -c "import sys; print(sys.version)"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -U pip

# Install EasyDeL (editable) with TPU extra.
python -m pip install -e ".[tpu]"

python -c "import jax; print('JAX devices:', jax.devices())"

python codex_grpo_gsm8k_tpuv6/train_grpo_gsm8k_qwen3_8b.py
