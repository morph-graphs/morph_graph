#!/usr/bin/env bash
# Install EGAgent's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4).
# Used by setup_env.sh (virtualenv) and, later, the Docker image, so both get the same stack.
#   bash install_requirements.sh <python>
set -euo pipefail
P=${1:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel setuptools
"$P" -m pip install -q -r "$HERE/requirements-client.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
