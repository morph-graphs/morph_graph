# ReKV environment image (morph_graph baselines/rekv) for EPIC-Morph runs off our clusters (e.g. Max's).
# One arm, m8 (Qwen3-VL-8B): the pinned environment of env/requirements.txt, the versions the vendored port
# (vendor/MemStream) was developed with: torch 2.6.0 (CUDA 12.6), transformers 5.0.0.
# The image holds the environment only. The code, the vendored port included, is the repo checkout, mounted at
# /repo at run time, so every run uses the commit that is checked out (run_rekv.sh records it in runs.jsonl).
#
#   docker build -f baselines/rekv/docker/Dockerfile --build-arg ARM=m8 -t rekv:m8 .   (from the morph_graph root)
#   (baselines/rekv/docker/build.sh does this with the published tag)
#
# Base: Python 3.11, torch 2.6.0, CUDA 12.6, cuDNN 9 (runtime; nothing is compiled at build time).
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python REKV_ARM=${ARM} REKV_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton HF_HOME=/hf

RUN test "$ARM" = m8 || { echo "build with --build-arg ARM=m8" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# ffmpeg to decode the video, git for runs.jsonl's commits, util-linux for flock (bank generation), gcc for
# Triton (the port's attention kernel is compiled at first use)
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg ca-certificates util-linux gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY baselines/rekv/env/ /tmp/rekv-env/
# into the base Python, which already has torch 2.6.0 (cu126), as env/setup_env.sh does in a virtualenv
RUN python -m pip install -q -r /tmp/rekv-env/requirements.txt && (python -m pip check || echo "pip check reports the conflicts above")

# no GPU at build time: check the versions, and that what the vendored port imports is there
RUN "$PY" -c "import torch, transformers, triton, einops, logzero, ezcolorlog, scipy, qwen_vl_utils; \
from transformers import Qwen3VLForConditionalGeneration; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'triton', triton.__version__); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '5.0.0', (torch.__version__, transformers.__version__)"

WORKDIR /repo
CMD ["/bin/bash"]
