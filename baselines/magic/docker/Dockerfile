# MAGIC-Video environment image (morph_graph baselines/magic) for EPIC-Morph runs off our clusters.
# One image per arm, both holding the client environment (MAGIC itself: memory build, chains, answering;
# transformers 4.57.6, which VLM2Vec-V2.0 needs):
#   m8   the client environment also serves the backbone (Qwen3-VL-8B)
#   m27  plus the m27 server environment in /opt/m27 (Qwen3.8-27B, transformers 5.8): MERIT's m27 stack,
#        installed by MERIT's script into a venv over the client's torch (--system-site-packages), so torch
#        and the CUDA libraries are not installed twice; the venv's own pins shadow the client's
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time
# (run_magic.sh records its commit in runs.jsonl); MAGIC itself is the submodule baselines/magic/MAGIC-video.
#
#   docker build -f baselines/magic/docker/Dockerfile --build-arg ARM=m8 -t magic:m8 .   (morph_graph root)
#   (baselines/magic/docker/build.sh does this; .github/workflows/baseline-images.yml builds and publishes)
#
# Base: the same as MERIT's image (Python 3.11, torch 2.6.0, CUDA 12.4, cuDNN 9; devel for m27's nvcc).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python MAGIC_ARM=${ARM} MAGIC_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton HF_HOME=/hf CUDA_HOME=/usr/local/cuda

RUN test "$ARM" = m8 -o "$ARM" = m27 || { echo "build with --build-arg ARM=m8 or ARM=m27" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# curl: run_magic.sh waits for the backbone server with it; ffmpeg: MERIT's prep (the clips)
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg curl libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the same installs as the virtualenvs get (env/setup_env.sh): prebuilt CUDA wheels, then the pins
COPY baselines/magic/env/ /tmp/magic-env/
COPY baselines/merit/env/ /tmp/merit-env/
RUN bash /tmp/magic-env/install_requirements.sh client python
RUN if [ "$ARM" = m27 ]; then python -m venv --system-site-packages /opt/m27 \
        && bash /tmp/merit-env/install_requirements.sh m27 /opt/m27/bin/python; fi
# run_magic.sh uses it for m27 only (m8 serves from PY)
ENV MAGIC_SERVER_PY=/opt/m27/bin/python

# no GPU at build time: check versions by metadata, and that the pinned stacks import
RUN python -c "import importlib.metadata as md, torch, transformers, peft, igraph, faiss, openai, sentence_transformers; \
from transformers import AutoModelForImageTextToText, Qwen2VLForConditionalGeneration; \
print('client: torch', torch.__version__, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '4.57.6', transformers.__version__"
RUN if [ "$ARM" = m27 ]; then /opt/m27/bin/python -c "import torch, transformers, causal_conv1d; \
from transformers import AutoModelForImageTextToText, AutoProcessor; \
print('m27 server: torch', torch.__version__, 'transformers', transformers.__version__); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '5.8.0', transformers.__version__"; fi

WORKDIR /repo
CMD ["/bin/bash"]
