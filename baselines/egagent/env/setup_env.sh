#!/usr/bin/env bash
# The EGAgent environment as a virtualenv: bash baselines/egagent/env/setup_env.sh <venv dir>
# One environment for every stage and both arms (the paper's LLMs are API models; SigLIP 2 runs locally).
# Needs Python 3.11 (PYTHON, default python3.11) and an NVIDIA driver for CUDA 12.4 wheels.
set -euo pipefail
DIR=${1:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3.11}" -m venv "$DIR"
P="$DIR/bin/python"
"$P" -m pip install -q --upgrade pip
# torch from PyTorch's index, its dependencies from PyPI (that index has some only as source archives)
"$P" -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
bash "$HERE/install_requirements.sh" "$P"
"$P" - <<'PY'
import torch, transformers, langchain, langgraph, langchain_openai, langchain_google_genai, langchain_experimental
from importlib.metadata import version
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "langchain", langchain.__version__, "langgraph", version("langgraph"), "GPUs", torch.cuda.device_count())
PY
