# Ego-R1 environment image (morph_graph baselines/egor1) for EPIC-Morph runs off our clusters (e.g. Max's):
# torch + transformers for the agent (Ego-R1-Agent-3B, ~8 GB of GPU), the OpenAI and Gemini clients and ffmpeg for
# upstream's tools. One image, tagged by arm (gemini25pro); arm llava (a LLaVA-Video server) has no image yet.
# The image holds the environment only: no keys, no weights. The code is the repo checkout, mounted at /repo at
# run time (the upstream code is the submodule baselines/egor1/Ego-R1, patched in place by run_egor1.sh).
#
#   docker build -f baselines/egor1/docker/Dockerfile --build-arg ARM=gemini25pro -t egor1:gemini25pro .   (morph_graph root)
#   (baselines/egor1/docker/build.sh does this, and .github/workflows/baseline-images.yml builds and publishes it)
#
# Base: as the other baselines' images (Python 3.11, torch 2.6.0, CUDA 12.4); nothing is compiled, so the runtime one.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG ARM=gemini25pro
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 PY=/opt/conda/bin/python EGOR1_IMAGE_COMMIT=${COMMIT} HOME=/tmp HF_HOME=/hf

RUN test "$ARM" = gemini25pro || { echo "build with --build-arg ARM=gemini25pro" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg patch util-linux libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the same pins as a virtualenv gets (env/setup_env.sh), over the base image's torch
COPY baselines/egor1/env/requirements.txt /tmp/egor1-env/requirements.txt
RUN pip install -r /tmp/egor1-env/requirements.txt

# no GPU at build time: the pinned stack imports, torch is still the base image's, and the two SDK methods the
# usage log wraps (epic/apis.py track) are where it expects them
RUN python -c "import importlib.metadata as md, torch, transformers, openai, cv2, ffmpeg, nltk, pandas, huggingface_hub; \
from google.genai import models as gm; from openai.resources.chat.completions import Completions; \
assert torch.__version__.startswith('2.6.0'), torch.__version__; assert transformers.__version__ == '4.57.6'; \
assert hasattr(gm.Models, 'generate_content') and hasattr(Completions, 'create'); \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'openai', openai.__version__, \
'google-genai', md.version('google-genai'))" \
    && ffmpeg -version | head -1 && patch --version | head -1

WORKDIR /repo
CMD ["/bin/bash"]
