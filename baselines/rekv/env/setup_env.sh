#!/usr/bin/env bash
# The virtualenv for ReKV on Qwen3-VL: bash baselines/rekv/env/setup_env.sh <venv dir>
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.6 wheels. ffmpeg and ffprobe
# must be on PATH when running (not installed here).
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q --upgrade pip
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
"$P" -m pip install -q -r "$HERE/requirements.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
"$P" -c 'import torch, transformers; print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "GPUs", torch.cuda.device_count())'
