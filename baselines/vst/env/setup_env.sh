#!/usr/bin/env bash
# The VST virtualenv: bash baselines/vst/env/setup_env.sh vst7b <venv dir>
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.4 wheels.
set -euo pipefail
ENV=${1:?vst7b}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
bash "$HERE/install_requirements.sh" "$ENV" "$P"
PYTHONPATH="$HERE/../VST/eval/lmms-eval" "$P" - <<'PY'
import torch, transformers, flash_attn, qwen_vl_utils.vision_process as q
from lmms_eval.models.simple.qwen2_5_vl_stream_think import Qwen2_5_VL_Stream_Think   # upstream's model class imports
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "qwen_vl_utils FPS", q.FPS, "GPUs", torch.cuda.device_count())
PY
