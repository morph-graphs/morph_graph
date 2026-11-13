# M3-Agent environment image (morph_graph baselines/m3agent) for EPIC-Morph runs off our clusters (e.g. Max's).
# One arm, m3 (the published checkpoints). It holds the two pinned environments env/setup_env.sh builds as
# virtualenvs (upstream needs two transformers versions that cannot share one):
#   /opt/m3-mem  memorization (prep, build): transformers at upstream's commit, flash-attn, insightface,
#                speakerlab (ERes2NetV2), faster-whisper; PY_MEM
#   /opt/m3-ctl  control (qa, blind, upload; the embedding model's helper process): vLLM 0.8.4, transformers
#                4.51.0, sentence-transformers; PY_CTL
# Both are virtualenvs over one Python (CPython 3.11.11, python-build-standalone via uv; no conda) that holds
# only torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0 (CUDA 12.4), shared through --system-site-packages
# so the ~6 GB torch stack is not installed twice; each venv's own pins shadow the base's.
# The image holds no model weights and no keys: models are a host folder mounted at /models (fetch_models.sh),
# API keys (M3AGENT_BACKEND=api) come from the environment at run time (docker/run.sh).
# The code is the repo checkout, mounted at /repo at run time (run_m3agent.sh records its commit in runs.jsonl);
# M3-Agent itself is the submodule baselines/m3agent/m3-agent (morph-graphs/m3-agent, branch epic-morph).
#
#   docker build -f baselines/m3agent/docker/Dockerfile --build-arg ARM=m3 -t m3agent:m3 .   (morph_graph root)
#   (baselines/m3agent/docker/build.sh does this; .github/workflows/baseline-images.yml builds and publishes)
#
# Base: CUDA 12.4 runtime with cuDNN 9 (CTranslate2, faster-whisper's backend, loads the system cuBLAS/cuDNN).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

ARG ARM=m3
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY_MEM=/opt/m3-mem/bin/python PY_CTL=/opt/m3-ctl/bin/python M3AGENT_ARM=${ARM} M3AGENT_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton NUMBA_CACHE_DIR=/tmp/numba HF_HOME=/hf \
    PYTHONNOUSERSITE=1
# PYTHONNOUSERSITE: apptainer mounts the host's home, whose ~/.local packages would otherwise shadow the pins

RUN test "$ARM" = m3 || { echo "build with --build-arg ARM=m3" >&2; exit 1; }

# git: the speakerlab clone and runs.jsonl's commits; a C compiler: insightface and hdbscan build from source,
# Triton (vLLM) compiles its launchers. No system ffmpeg: install_requirements.sh links imageio-ffmpeg's static
# build into each venv, as in the venvs the port was checked with.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the base Python and the shared torch (as setup_env.sh installs it into each venv), at a fixed path: the venvs'
# pyvenv.cfg points at its folder (a symlink to the interpreter alone would not find the standard library). uv
# marks its Pythons externally managed (PEP 668), which blocks even ensurepip: this one is the image's own base,
# so the marker goes
ENV PYBASE=/opt/python/base/bin/python3.11
RUN UV_PYTHON_INSTALL_DIR=/opt/python uv python install 3.11.11 \
    && ln -s "$(dirname "$(dirname "$(UV_PYTHON_INSTALL_DIR=/opt/python uv python find --python-preference only-managed 3.11.11)")")" \
        /opt/python/base \
    && rm -f "$("$PYBASE" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')/EXTERNALLY-MANAGED" \
    && "$PYBASE" -m ensurepip \
    && "$PYBASE" -m pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY baselines/m3agent/env/ /tmp/m3agent-env/
RUN "$PYBASE" -m venv --system-site-packages /opt/m3-mem && bash /tmp/m3agent-env/install_requirements.sh mem "$PY_MEM"
RUN "$PYBASE" -m venv --system-site-packages /opt/m3-ctl && bash /tmp/m3agent-env/install_requirements.sh ctl "$PY_CTL"

# no GPU at build time: check versions by metadata, and that each pinned stack imports (setup_env.sh's checks)
RUN "$PY_MEM" -c "import importlib.metadata as md, torch, transformers, insightface, hdbscan, faster_whisper, speakerlab, qwen_omni_utils; \
from transformers import Qwen2_5OmniThinkerForConditionalGeneration; \
print('mem: torch', torch.__version__, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn')); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__.startswith('4.50.0.dev'), transformers.__version__" \
    && "$PY_CTL" -c "import importlib.metadata as md, torch, transformers, sentence_transformers, openai; \
print('ctl: torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', md.version('vllm')); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '4.51.0' and md.version('vllm') == '0.8.4'" \
    && /opt/m3-mem/bin/ffmpeg -version | head -1

WORKDIR /repo
CMD ["/bin/bash"]
