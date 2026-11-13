#!/usr/bin/env bash
# One virtualenv per MAGIC environment: bash baselines/magic/env/setup_env.sh <client|m27> <venv dir>
#   client  MAGIC itself (memory build, answering) for both arms, and the m8 backbone server
#   m27     the m27 backbone server only (Qwen3.8-27B): MERIT's m27 environment
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.4 wheels; m27 compiles
# causal-conv1d (CUDA_HOME with nvcc, ~10 min).
set -euo pipefail
ENV=${1:?client or m27}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
bash "$HERE/install_requirements.sh" "$ENV" "$P"
"$P" - <<'PY'
import torch, transformers, flash_attn
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "GPUs", torch.cuda.device_count())
PY
