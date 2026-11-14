#!/usr/bin/env bash
# The EgoRAG environment (EgoGPT + EgoRAG in one virtualenv): bash baselines/egorag/env/setup_env.sh <venv dir>
# Needs Python 3.10 or 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.1 wheels. At run time:
# ffmpeg and patch on PATH, and CUDA_HOME (a CUDA 12 toolkit with nvcc) for the captions.
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
bash "$HERE/install_requirements.sh" "$P"
PYTHONPATH="$HERE/../EgoLife/EgoGPT:$HERE/../EgoLife/EgoRAG:$HERE/../epic" "$P" "$HERE/check_env.py"
