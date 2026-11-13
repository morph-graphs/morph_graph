# WorldMM environment image (morph_graph baselines/worldmm) for EPIC-Morph runs off our clusters (e.g. Max's).
# One image per arm, holding the same pinned environments as env/setup_env.sh builds as virtualenvs:
#   m8   the client environment (WorldMM itself + the Qwen3-VL-8B server), transformers 4.57.6
#   m27  the client environment, plus the server27 environment (the Qwen3.8-27B server, transformers 5.8.0)
#        as a virtualenv at /opt/server
# PY (WorldMM) and SERVER_PY (the backbone server) point at them; in m8 both are the same Python.
# The image holds the environments only. The code is the repo checkout, mounted at /repo at run time, so every
# run uses the commit that is checked out (run_worldmm.sh records it in runs.jsonl); WorldMM itself is the
# submodule baselines/worldmm/WorldMM (morph-graphs/WorldMM, branch epic-morph), put on the path by the scripts.
#
#   docker build -f baselines/worldmm/docker/Dockerfile --build-arg ARM=m8 -t worldmm:m8 .   (morph_graph root)
#   (baselines/worldmm/docker/build.sh does this with the published tag)
#
# Base: MERIT's (Python 3.11, torch 2.6.0, CUDA 12.4, cuDNN 9; devel, for nvcc to compile causal-conv1d).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python SERVER_PY=/opt/server/bin/python WORLDMM_ARM=${ARM} WORLDMM_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton HF_HOME=/hf CUDA_HOME=/usr/local/cuda

RUN test "$ARM" = m8 -o "$ARM" = m27 || { echo "build with --build-arg ARM=m8 or ARM=m27" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# ffmpeg for the prep stage, git for runs.jsonl's commits, util-linux for flock (bank generation)
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY baselines/worldmm/env/ /tmp/worldmm-env/
# client: into the base Python, which already has torch 2.6.0 (cu124), as env/install_requirements.sh expects
RUN bash /tmp/worldmm-env/install_requirements.sh client python
# server: m27 gets its own virtualenv exactly as env/setup_env.sh makes it (torch from the cu124 index, then
# the pins and the compiled causal-conv1d); m8's server is the client Python
RUN if [ "$ARM" = m27 ]; then PYTHON=/opt/conda/bin/python bash /tmp/worldmm-env/setup_env.sh server27 /opt/server; \
    else mkdir -p /opt/server/bin && ln -s /opt/conda/bin/python /opt/server/bin/python; fi

# no GPU at build time: check versions by metadata, and that each pinned stack imports
RUN "$PY" -c "import importlib.metadata as md, torch, transformers, sentence_transformers, decord, igraph, peft; \
print('client: torch', torch.__version__, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '4.57.6', transformers.__version__" \
    && "$SERVER_PY" -c "import importlib.metadata as md, torch, transformers, qwen_vl_utils; \
print('server: torch', torch.__version__, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0'); \
assert transformers.__version__ == {'m8': '4.57.6', 'm27': '5.8.0'}['$ARM'], transformers.__version__"

WORKDIR /repo
CMD ["/bin/bash"]
