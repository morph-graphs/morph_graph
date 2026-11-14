#!/usr/bin/env bash
# The stack under every baseline's arm64 image (docker/Dockerfile.arm64): what the x86 images get from their base
# (pytorch/pytorch:2.6.0-cuda12.4) and from prebuilt x86 wheels, rebuilt for aarch64 hosts (Grace: GH200, GB200,
# GB300). Run as the first step on nvidia/cuda:12.9.1-devel-ubuntu22.04; the layers it makes are the same in every
# baseline's image, so a local build compiles flash-attn once.
#   Python 3.11 (deadsnakes) in a virtualenv at /opt/py, as the x86 images' /opt/conda/bin/python
#   torch 2.8.0 + torchvision 0.23.0 for CUDA 12.9: torch 2.6 has no aarch64 CUDA wheels; 2.8 is the first built with
#     CUDA 12.9 (Blackwell, GB300's sm_103 included). Its kernels are sm_80/90/100/120; sm_100 ones also run on sm_103
#   A virtualenv over /opt/py (MAGIC's and WorldMM's m27 servers) gets /opt/py's site-packages from a .pth file, after
#     its own: venv does not nest, so --system-site-packages (x86's way) would point at the system Python instead
#   /wheels: flash-attn 2.7.4.post1 (the x86 version) and decord 0.6.0, compiled here (neither publishes aarch64
#     wheels); each baseline's env/wheels-<env>-arm64.txt installs them as wheels-<env>.txt does on x86
# MAX_JOBS bounds the parallel nvcc jobs: flash-attn's largest take ~7 GB of RAM each (measured on arm64), so 3 for a
# 16 GB runner (with swap), 16-32 on a big machine.
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends software-properties-common gpg-agent
add-apt-repository -y ppa:deadsnakes/ppa
# the baselines' runtime tools (git for runs.jsonl, ffmpeg for the clips, curl for server waits, util-linux for
# flock), python3.11-dev for Triton's and the extensions' Python.h, and decord's build dependencies
apt-get install -y --no-install-recommends python3.11-dev python3.11-venv git ffmpeg curl libgl1 libglib2.0-0 \
    ca-certificates util-linux build-essential cmake libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev \
    libswscale-dev libswresample-dev libavutil-dev
rm -rf /var/lib/apt/lists/*
git config --system --add safe.directory '*'

python3.11 -m venv /opt/py
P=/opt/py/bin/python
"$P" -m pip install -q --upgrade pip wheel setuptools packaging ninja psutil
"$P" -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
"$P" -m pip install -q einops   # flash-attn's setup imports it

# decord 0.6.0 (CPU decoding, as the x86 wheel) against the system FFmpeg 4.4; its setup.py puts
# libdecord.so inside the package. Before flash-attn, the long step, so that a failure here shows early
git clone -q --recursive --depth 1 --branch v0.6.0 https://github.com/dmlc/decord /tmp/decord
cmake -S /tmp/decord -B /tmp/decord/build -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release >/dev/null
make -C /tmp/decord/build -j"${MAX_JOBS:-3}" >/dev/null
(cd /tmp/decord/python && "$P" -m pip wheel -q --no-deps . -w /wheels)

# flash-attn: the same release as x86, kernels for sm_80 (always built), sm_90 and sm_100; the sm_100 cubins also
# run on sm_103 (GB300). FORCE_BUILD skips the attempt to download an x86 wheel.
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="90;100" MAX_JOBS="${MAX_JOBS:-3}" NVCC_THREADS=1 \
    "$P" -m pip wheel -q --no-build-isolation --no-deps flash-attn==2.7.4.post1 -w /wheels
rm -rf /tmp/decord /tmp/pip-* /root/.cache
ls -l /wheels
