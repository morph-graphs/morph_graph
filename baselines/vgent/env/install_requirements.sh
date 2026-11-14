#!/usr/bin/env bash
# Install Vgent's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4): first the prebuilt
# flash-attn wheel (wheels.txt), then requirements.txt. Used by setup_env.sh (and a Docker image later), so
# both get the same stack.
#   bash install_requirements.sh <python>
set -euo pipefail
P=${1:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel setuptools
"$P" -m pip install -q $(grep -v '^#' "$HERE/wheels.txt")
"$P" -m pip install -q -r "$HERE/requirements.txt"
# pip check calls decord 0.6.0 "not supported on this platform" (its wheel's old manylinux tag); it imports and reads
"$P" -m pip check || echo "pip check reports the above (expected only for decord)"
