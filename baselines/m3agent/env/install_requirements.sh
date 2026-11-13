#!/usr/bin/env bash
# Install one M3-Agent environment's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4):
#   bash install_requirements.sh <mem|ctl> <python>
# mem: memorization (transformers at upstream's commit, flash-attn, insightface, ERes2NetV2 via 3D-Speaker's
#      speakerlab, faster-whisper); ctl: control (vLLM 0.8.4, transformers 4.51.0) and the embedding model.
# Used by setup_env.sh (virtualenv); a Docker image can call it the same way.
set -euo pipefail
ENV=${1:?mem or ctl}; P=${2:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel setuptools packaging ninja Cython   # build tools (insightface, hdbscan)
[ -s "$HERE/wheels-$ENV.txt" ] && "$P" -m pip install -q --no-build-isolation $(grep -v '^#' "$HERE/wheels-$ENV.txt")
# requirements-<env>.txt is the complete set (every package and version), so it installs without resolving
"$P" -m pip install -q --no-deps -r "$HERE/requirements-$ENV.txt"
BIN=$(dirname "$P")
if [ "$ENV" = mem ]; then
    # speakerlab (the ERes2NetV2 speaker-embedding code) is not on PyPI: upstream copies it from 3D-Speaker.
    # Pinned commit; the package is copied into site-packages.
    SP=$("$P" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    TMP=$(mktemp -d); git clone -q https://github.com/modelscope/3D-Speaker "$TMP/3D-Speaker"
    git -C "$TMP/3D-Speaker" checkout -q 065629c313eaf1a01c65c640c46d77e61e9607b4
    rm -rf "$SP/speakerlab"; cp -r "$TMP/3D-Speaker/speakerlab" "$SP/speakerlab"; rm -rf "$TMP"
fi
# ffmpeg on the environment's PATH (moviepy, audioread/librosa and our clip cutting call it): imageio-ffmpeg's
# static build, unless the system has one
if ! command -v ffmpeg >/dev/null && [ ! -e "$BIN/ffmpeg" ] && "$P" -c "import imageio_ffmpeg" 2>/dev/null; then
    ln -s "$("$P" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")" "$BIN/ffmpeg"
fi
"$P" -m pip check || echo "pip check reports the conflicts above (expected: vllm 0.8.4 asks for transformers >= 4.51.1, upstream installs 4.51.0 over it)"
