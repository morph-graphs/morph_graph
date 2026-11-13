# API-baselines environment image (morph_graph baselines/api) for EPIC-Morph runs off our clusters (e.g. Max's):
# the Gemini and OpenAI clients, huggingface_hub for the results upload, and ffmpeg. No GPU, no model weights.
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time, so every
# run uses the commit on main that is checked out (recorded in runs.jsonl).
#
#   docker build -f baselines/api/docker/Dockerfile --build-arg ARM=cpu -t api:cpu .   (morph_graph root)
#   (baselines/api/docker/build.sh does this, and .github/workflows/baseline-images.yml builds and publishes it)
FROM python:3.11-slim-bookworm

ARG ARM=cpu
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    PY=python API_IMAGE_COMMIT=${COMMIT} HOME=/tmp HF_HOME=/hf

RUN test "$ARM" = cpu || { echo "build with --build-arg ARM=cpu" >&2; exit 1; }

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY baselines/api/env/requirements.txt /tmp/api-env/requirements.txt
RUN pip install -r /tmp/api-env/requirements.txt

# the pinned clients import, and ffmpeg knows the option the frame reader uses (ffmpeg >= 5.1)
RUN python -c "import importlib.metadata as md, google.genai, openai, huggingface_hub; \
print('google-genai', md.version('google-genai'), 'openai', openai.__version__, 'huggingface_hub', huggingface_hub.__version__)" \
    && ffmpeg -hide_banner -h full | grep -q -- "-fps_mode" && ffmpeg -version | head -1

WORKDIR /repo
CMD ["/bin/bash"]
