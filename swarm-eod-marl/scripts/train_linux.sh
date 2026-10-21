#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export ARCADE_HEADLESS=True
export PYTHONPATH="${project_dir}/../swarm-eod-main/src:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install a compatible PyTorch CUDA wheel and check nvidia-smi.")
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
PY

exec python -m swarm_marl.train \
  --config configs/train_10bombs.yaml \
  --run-dir runs/ten_bomb_training \
  --device cuda "$@"
