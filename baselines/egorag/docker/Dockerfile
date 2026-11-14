# EgoRAG environment image (morph_graph baselines/egorag) for EPIC-Morph runs off our clusters (e.g. Max's).
# One image, arm egogpt_gpt4o: EgoGPT (captions) and EgoRAG (index, summaries, answers) in one Python, the same
# pinned stack as env/setup_env.sh builds as a virtualenv.
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time, so every
# run uses the commit that is checked out (run_egorag.sh records it in runs.jsonl); EgoLife itself is the
# submodule baselines/egorag/EgoLife (upstream, pinned), which run_egorag.sh patches with EgoLife.patch.
#
#   docker build -f baselines/egorag/docker/Dockerfile --build-arg ARM=egogpt_gpt4o -t egorag:egogpt_gpt4o .
#   (from the morph_graph root; baselines/egorag/docker/build.sh does this with the published tag)
#
# Base: EgoGPT's pins (torch 2.1.2, CUDA 12.1), Python 3.10. devel, because deepspeed (imported by EgoGPT's
# Whisper encoder) refuses to import on a GPU machine without a CUDA toolkit: CUDA_HOME with nvcc.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG ARM=egogpt_gpt4o
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python EGORAG_ARM=${ARM} EGORAG_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba HF_HOME=/hf CUDA_HOME=/usr/local/cuda

RUN test "$ARM" = egogpt_gpt4o || { echo "build with --build-arg ARM=egogpt_gpt4o" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version" \
    && test -x "$CUDA_HOME/bin/nvcc"

# ffmpeg for the clips and for librosa reading a clip's audio, patch for EgoLife.patch, git for runs.jsonl's
# commits, util-linux for flock
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg patch libgl1 libglib2.0-0 \
        ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the same install as a virtualenv gets (env/setup_env.sh): into the base Python, which has torch, torchvision and torchaudio 2.1.2 (cu121)
COPY baselines/egorag/env/ /tmp/egorag-env/
RUN bash /tmp/egorag-env/install_requirements.sh python

# no GPU at build time: the pinned stack imports (deepspeed with CUDA_HOME set), and Chroma gets a new enough
# sqlite through pysqlite3 (what epic/build_memory.py swaps in)
RUN python -c "import sys, pysqlite3; sys.modules['sqlite3'] = pysqlite3; \
import chromadb, decord, deepspeed, librosa, open_clip, openai, torch, torchaudio, transformers, whisper; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'chromadb', chromadb.__version__, \
'openai', openai.__version__, 'sqlite', pysqlite3.sqlite_version); \
assert torch.__version__.startswith('2.1.2') and transformers.__version__ == '4.45.2'; \
assert chromadb.__version__ == '0.5.20' and openai.__version__ == '1.55.3'; \
assert tuple(map(int, pysqlite3.sqlite_version.split('.'))) >= (3, 35)"

WORKDIR /repo
CMD ["/bin/bash"]
