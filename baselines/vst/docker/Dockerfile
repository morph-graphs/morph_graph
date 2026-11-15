# VST environment image (morph_graph baselines/vst) for EPIC-Morph runs off our clusters (e.g. Max's).
# One arm, vst7b (Catalan258/VST-7B, Qwen2.5-VL-7B weights): the pinned environment of env/ (wheels-vst7b.txt,
# requirements-vst7b.txt, installed by env/install_requirements.sh as in a virtualenv): torch 2.6.0 (CUDA 12.4),
# transformers 4.52.3, flash-attn 2.7.4.post1 (a prebuilt wheel).
# The image holds the environment only. The code is the repo checkout, mounted at /repo at run time, so every run
# uses the commit that is checked out (run_vst.sh records it in runs.jsonl); upstream's inference code is the
# submodule baselines/vst/VST of that checkout. The model is not in the image: it comes from the mounted
# Hugging Face cache (16.6 GB, a public download).
#
#   docker build -f baselines/vst/docker/Dockerfile --build-arg ARM=vst7b -t vst:vst7b .   (from the morph_graph root)
#   (baselines/vst/docker/build.sh does this with the published tag)
#
# Base: Python 3.11, torch 2.6.0, CUDA 12.4, cuDNN 9 (runtime; nothing is compiled at build time).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG ARM
ARG COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    PY=/opt/conda/bin/python VST_ARM=${ARM} VST_IMAGE_COMMIT=${COMMIT} \
    HOME=/tmp HF_HOME=/hf

RUN test "$ARM" = vst7b || { echo "build with --build-arg ARM=vst7b" >&2; exit 1; } \
    && python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# git for runs.jsonl's commits, util-linux for flock (bank generation). The video is read by decord and PyAV,
# whose wheels carry their own FFmpeg libraries
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# the same install as a virtualenv gets (env/setup_env.sh), into the base Python, which already has torch 2.6.0 (cu124)
COPY baselines/vst/env/ /tmp/vst-env/
RUN bash /tmp/vst-env/install_requirements.sh "$ARM" python

# no GPU at build time: check the versions (flash-attn by metadata), and that what the adapter and upstream's
# model class import is there. The class itself is in the submodule, which is mounted at run time, not built in.
RUN "$PY" -c "import importlib.metadata as md, torch, torchvision, transformers, accelerate, decord, av, loguru, sqlitedict, evaluate, datasets, yaml, jinja2, pytz, tqdm; \
import qwen_vl_utils.vision_process as q; \
from transformers import Qwen2_5_VLForConditionalGeneration; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'flash-attn', md.version('flash_attn'), 'qwen_vl_utils FPS', q.FPS); \
assert torch.__version__.startswith('2.6.0') and torch.version.cuda == '12.4', (torch.__version__, torch.version.cuda); \
assert transformers.__version__ == '4.52.3' and md.version('flash_attn') == '2.7.4.post1', (transformers.__version__, md.version('flash_attn'))"

WORKDIR /repo
CMD ["/bin/bash"]
