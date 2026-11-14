# EGAgent environment image (morph_graph baselines/egagent) for EPIC-Morph runs off our clusters (e.g. Max's).
# One image for both arms (gpt41, gemini25pro): the arms differ only in the API model the agent calls. It holds
# the pinned environment of env/setup_env.sh (the LangChain / LangGraph stack and the API clients, and torch +
# transformers for the local SigLIP 2 embedder) and ffmpeg. No model weights, no keys.
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time, so every
# run uses the commit that is checked out (run_egagent.sh records it in runs.jsonl); EGAgent itself is the
# submodule baselines/egagent/egagent (upstream at its pinned commit), patched in place by run_egagent.sh.
#
#   docker build -f baselines/egagent/docker/Dockerfile --build-arg ARM=gpu -t egagent:gpu .   (morph_graph root)
#   (baselines/egagent/docker/build.sh does this with the published tag)
#
# Base: Python 3.11, torch 2.6.0, CUDA 12.4, cuDNN 9 (runtime: nothing is compiled).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG ARM=gpu
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python EGAGENT_IMAGE_COMMIT=${COMMIT} HOME=/tmp HF_HOME=/hf

RUN test "$ARM" = gpu || { echo "build with --build-arg ARM=gpu" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# ffmpeg for the prep stage, git for runs.jsonl's commits and the patch, util-linux for flock
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY baselines/egagent/env/ /tmp/egagent-env/
# into the base Python, which already has torch 2.6.0 (cu124), as env/install_requirements.sh expects
RUN bash /tmp/egagent-env/install_requirements.sh python

# no GPU at build time: check versions by metadata, and that the pinned stack imports
RUN "$PY" -c "import importlib.metadata as md, torch, transformers, langgraph, langchain_openai, langchain_google_genai, \
langchain_experimental, bm25s, pysrt, huggingface_hub; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'langgraph', md.version('langgraph'), \
'langchain-openai', md.version('langchain-openai'), 'langchain-google-genai', md.version('langchain-google-genai')); \
assert torch.__version__.startswith('2.6.0') and transformers.__version__ == '4.53.2', transformers.__version__" \
    && ffmpeg -version | head -1

WORKDIR /repo
CMD ["/bin/bash"]
