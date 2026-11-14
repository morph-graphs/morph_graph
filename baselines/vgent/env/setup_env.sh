#!/usr/bin/env bash
# The Vgent virtualenv (one for every arm: the arms are Vgent's own backbones, all Qwen-VL):
#   bash baselines/vgent/env/setup_env.sh <venv dir>
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.4 wheels.
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
bash "$HERE/install_requirements.sh" "$P"
"$P" - <<'PY'
import torch, transformers, flash_attn, decord, networkx
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "decord", decord.__version__, "GPUs", torch.cuda.device_count())
PY
