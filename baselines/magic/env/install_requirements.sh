#!/usr/bin/env bash
# Install one MAGIC environment's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4).
# Used by setup_env.sh (virtualenv) and later by a Docker image, so both get the same stack.
#   bash install_requirements.sh <client|m27> <python>
#   client  MAGIC's own process (both arms) and the m8 server (Qwen3-VL-8B): transformers 4.57
#   m27     the m27 server (Qwen3.8-27B, transformers 5.8): exactly MERIT's m27 stack, installed by MERIT's script
set -euo pipefail
ENV=${1:?client or m27}; P=${2:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $ENV in
    m27) exec bash "$HERE/../../merit/env/install_requirements.sh" m27 "$P" ;;
    client) ;;
    *) echo "environment must be client or m27"; exit 1 ;;
esac
"$P" -m pip install -q --upgrade pip wheel setuptools packaging ninja   # build tools for --no-build-isolation
"$P" -m pip install -q --no-build-isolation $(grep -v '^#' "$HERE/wheels-client.txt")
"$P" -m pip install -q -r "$HERE/requirements-client.txt"
"$P" -m pip check || echo "pip check reports the conflicts above"
