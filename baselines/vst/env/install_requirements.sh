#!/usr/bin/env bash
# Install the VST environment's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4).
# Kept apart from setup_env.sh so a Docker image can run the same install.
#   bash install_requirements.sh vst7b <python>
set -euo pipefail
ENV=${1:?vst7b}; P=${2:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ "$ENV" = vst7b ] || { echo "environment must be vst7b"; exit 1; }
"$P" -m pip install -q --upgrade pip wheel setuptools packaging
"$P" -m pip install -q $(grep -v '^#' "$HERE/wheels-$ENV.txt")
"$P" -m pip install -q -r "$HERE/requirements-$ENV.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
