#!/usr/bin/env bash
# The virtualenv of the Ego-R1 port: bash baselines/egor1/env/setup_env.sh <venv dir>
# Needs Python 3.11 (PYTHON, default python3.11), an NVIDIA driver for CUDA 12.4 wheels, and ffmpeg on PATH at
# run time. Arm `llava` also needs upstream's LLaVA-Video server in an environment of its own (README.md).
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q --upgrade pip
"$P" -m pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$P" -m pip install -q -r "$HERE/requirements.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
"$P" - <<'PY'
import torch, transformers, openai, google.genai, ffmpeg, cv2, nltk, pandas
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "openai", openai.__version__, "google-genai", google.genai.__version__, "GPUs", torch.cuda.device_count())
PY
