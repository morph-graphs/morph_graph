#!/usr/bin/env bash
# One virtualenv per M3-Agent stack (upstream needs two conflicting transformers versions):
#   bash baselines/m3agent/env/setup_env.sh <mem|ctl> <venv dir>
# Needs Python 3.11 (PYTHON, default python3.11), git, a C compiler (insightface and hdbscan build from source)
# and an NVIDIA driver for CUDA 12.4 wheels. Then PY_MEM=<mem dir>/bin/python PY_CTL=<ctl dir>/bin/python.
set -euo pipefail
ENV=${1:?mem or ctl}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
bash "$HERE/install_requirements.sh" "$ENV" "$P"
if [ "$ENV" = mem ]; then
    "$P" -c "import torch, transformers, flash_attn, insightface, hdbscan, faster_whisper, speakerlab, qwen_omni_utils
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
print('ok mem: torch', torch.__version__, 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"
else
    "$P" -c "import torch, transformers, vllm, sentence_transformers
print('ok ctl: torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
fi
