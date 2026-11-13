#!/usr/bin/env bash
# The EgoRAG environment (EgoGPT + EgoRAG in one virtualenv): bash baselines/egorag/env/setup_env.sh <venv dir>
# Needs Python 3.10 or 3.11 (PYTHON, default python3.11), an NVIDIA driver for CUDA 12.1 wheels, and ffmpeg on
# PATH at run time (clips, and librosa reading a clip's audio).
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q --upgrade pip wheel "setuptools<81"
"$P" -m pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
"$P" -m pip install -q --no-build-isolation -r "$HERE/requirements-nobuildiso.txt"
"$P" -m pip install -q -r "$HERE/requirements.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
PYTHONPATH="$HERE/../EgoLife/EgoGPT:$HERE/../EgoLife/EgoRAG" "$P" - <<'PY'
import sys, pysqlite3
sys.modules["sqlite3"] = pysqlite3   # as epic/build_memory.py does
import torch, transformers, chromadb, open_clip, whisper, deepspeed
from egogpt.model.builder import load_pretrained_model
from egorag.agents.RagAgent import RagAgent
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "chromadb", chromadb.__version__, "open_clip", open_clip.__version__, "GPUs", torch.cuda.device_count())
PY
