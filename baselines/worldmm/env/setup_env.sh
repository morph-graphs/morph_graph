#!/usr/bin/env bash
# One virtualenv per WorldMM environment: bash baselines/worldmm/env/setup_env.sh <client|server27> <venv dir>
#   client    WorldMM itself, and the backbone server of arm m8 (Qwen3-VL-8B)        transformers 4.57.6
#   server27  the backbone server of arm m27 (Qwen3.8-27B); WorldMM runs in client   transformers 5.8.0
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.4 wheels; server27 also nvcc
# (CUDA_HOME) to compile causal-conv1d.
set -euo pipefail
ENV=${1:?client or server27}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
bash "$HERE/install_requirements.sh" "$ENV" "$P"
"$P" - <<'PY'
import torch, transformers, flash_attn
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "GPUs", torch.cuda.device_count())
PY
