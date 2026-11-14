#!/usr/bin/env bash
# Install the EgoRAG environment's pinned packages into a Python that already has torch 2.1.2 (CUDA 12.1).
# Used by setup_env.sh (virtualenv) and docker/Dockerfile (image), so both get the same stack.
#   bash install_requirements.sh <python>
set -euo pipefail
P=${1:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel "setuptools<81"
"$P" -m pip install -q --no-build-isolation -r "$HERE/requirements-nobuildiso.txt"
"$P" -m pip install -q -r "$HERE/requirements.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
