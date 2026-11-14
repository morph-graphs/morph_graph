#!/usr/bin/env bash
# One virtualenv per HippoMM environment: bash baselines/hippomm/env/setup_env.sh <client|vllm> <venv dir>
#   client  HippoMM itself (ImageBind, Whisper, memory build, answering): upstream's pins, torch 2.1.2
#   vllm    the Qwen2.5-VL-7B-Instruct endpoint
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12 wheels.
set -euo pipefail
ENV=${1:?client or vllm}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
if [ "$ENV" = client ]; then
    "$P" -m pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi
bash "$HERE/install_requirements.sh" "$ENV" "$P"
if [ "$ENV" = client ]; then
    "$P" - <<'PY'
import torch, ctranslate2, faster_whisper, imagebind, token_count
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "ctranslate2", ctranslate2.__version__,
      "faster_whisper", faster_whisper.__version__, "GPUs", torch.cuda.device_count())
PY
else
    "$P" -c "import vllm, torch, transformers; print('ok: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
fi
"$P" -m pip freeze > "$DIR/freeze.txt"
