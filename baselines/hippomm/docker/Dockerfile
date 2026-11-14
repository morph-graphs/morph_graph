# HippoMM environment image (morph_graph baselines/hippomm) for EPIC-Morph runs off our clusters (e.g. Max's).
# One image for the one arm (q25vl7b-gpt4o), holding the same pinned environments as env/setup_env.sh builds:
#   vllm    the Qwen2.5-VL-7B-Instruct endpoint (vLLM 0.8.5.post1), in the base Python, which has its torch 2.6.0
#   client  HippoMM itself (ImageBind, Whisper, memory build, answering): upstream's pins on torch 2.1.2, as a
#           virtualenv at /opt/client (its torch wheel carries CUDA 12.1 and cuDNN 8)
# PY (HippoMM) and HIPPOMM_VLLM_PY (the endpoint) point at them.
# The image holds the environments only. The code is the repo checkout, mounted at /repo at run time, so every
# run uses the commit that is checked out (run_hippomm.sh records it in runs.jsonl); HippoMM itself is the
# submodule baselines/hippomm/HippoMM (upstream), patched in place by run_hippomm.sh with HippoMM.patch.
# No model weights and no key are in it: models come from the mounted Hugging Face cache, OPENAI_API_KEY from
# the environment at run time.
#
#   docker build -f baselines/hippomm/docker/Dockerfile --build-arg ARM=q25vl7b-gpt4o -t hippomm:q25vl7b-gpt4o .
#   (baselines/hippomm/docker/build.sh does this with the published tag)
#
# Base: Python 3.11, torch 2.6.0, CUDA 12.4 (runtime: nothing is compiled).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/client/bin/python HIPPOMM_VLLM_PY=/opt/conda/bin/python HIPPOMM_ARM=${ARM} HIPPOMM_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp HF_HOME=/hf

RUN test "$ARM" = q25vl7b-gpt4o || { echo "build with --build-arg ARM=q25vl7b-gpt4o" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# ffmpeg/ffprobe for frame and audio extraction, git for pip's git+ requirements and runs.jsonl's commits,
# patch for HippoMM.patch, curl for the endpoint check and the ImageBind weights, util-linux for flock
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg patch curl libgl1 libglib2.0-0 \
        libsndfile1 ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY baselines/hippomm/env/ /tmp/hippomm-env/
# vllm: into the base Python, which already has the torch 2.6.0 it pins
RUN bash /tmp/hippomm-env/install_requirements.sh vllm python
# client: its own virtualenv exactly as env/setup_env.sh makes it (torch 2.1.2 from the cu121 index, then the pins)
RUN PYTHON=/opt/conda/bin/python bash /tmp/hippomm-env/setup_env.sh client /opt/client

# no GPU at build time: check versions by metadata, and that each pinned stack imports
RUN "$PY" -c "import torch, ctranslate2, faster_whisper, imagebind, token_count, cv2, openai, tiktoken; \
print('client: torch', torch.__version__, 'ctranslate2', ctranslate2.__version__, 'faster_whisper', faster_whisper.__version__); \
assert torch.__version__.startswith('2.1.2') and faster_whisper.__version__ == '1.0.3'" \
    && "$HIPPOMM_VLLM_PY" -c "import importlib.metadata as md, torch, transformers; \
print('vllm:', md.version('vllm'), 'torch', torch.__version__, 'transformers', transformers.__version__); \
assert md.version('vllm') == '0.8.5.post1' and torch.__version__.startswith('2.6.0') and transformers.__version__ == '4.51.3'"

WORKDIR /repo
CMD ["/bin/bash"]
