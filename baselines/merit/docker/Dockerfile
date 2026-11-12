# MERIT environment image (morph_graph baselines/merit) for EPIC-Morph runs off our clusters (e.g. Max's).
# One image per arm: m8 (Qwen3-VL-8B, transformers 4.57.3) or m27 (Qwen3.8-27B, transformers 5.8.0).
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time, so
# every run uses the commit on main that is checked out (run_merit.sh records it in runs.jsonl). The MERIT
# code itself is the submodule baselines/merit/MERIT (morph-graphs/MERIT, branch epic-morph).
#
#   docker build -f baselines/merit/docker/Dockerfile --build-arg ARM=m27 -t merit:m27 .   (morph_graph root)
#   (from the repo root; baselines/merit/docker/build.sh does this, and .github/workflows/merit-images.yml
#   builds and publishes both arms)
#
# Base: the upstream MERIT image's (Python 3.11, torch 2.6.0, CUDA 12.4, cuDNN 9).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python MERIT_ARM=${ARM} MERIT_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton HF_HOME=/hf CUDA_HOME=/usr/local/cuda

RUN test "$ARM" = m8 -o "$ARM" = m27 || { echo "build with --build-arg ARM=m8 or ARM=m27" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the same install as a virtualenv gets (env/setup_env.sh): prebuilt CUDA wheels, then the pins
COPY baselines/merit/env/ /tmp/merit-env/
RUN bash /tmp/merit-env/install_requirements.sh "$ARM" python

# no GPU at build time: check versions by metadata, and that the pinned stack imports
RUN python -c "import importlib.metadata as md, torch, transformers, accelerate, sentence_transformers; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0'); \
assert transformers.__version__ == {'m8': '4.57.3', 'm27': '5.8.0'}['$ARM'], transformers.__version__"

WORKDIR /repo
CMD ["/bin/bash"]
