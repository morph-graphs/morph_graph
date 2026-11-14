#!/usr/bin/env bash
# Install one HippoMM environment's pinned packages. Used by setup_env.sh (and later the Docker image).
#   bash install_requirements.sh <client|vllm> <python>
#   client  HippoMM's own process; <python> already has torch 2.1.2 (CUDA 12.1)
#   vllm    the Qwen2.5-VL-7B endpoint (vLLM brings its own torch)
set -euo pipefail
ENV=${1:?client or vllm}; P=${2:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel
case $ENV in
    vllm) "$P" -m pip install -q -r "$HERE/requirements-vllm.txt" ;;
    client)
        "$P" -m pip install -q -r "$HERE/requirements-client.txt"
        # ImageBind alone: its requirements name another pytorchvideo commit than upstream HippoMM's pin
        "$P" -m pip install -q --no-deps "git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be" ;;
    *) echo "environment must be client or vllm"; exit 1 ;;
esac
"$P" -m pip check || echo "pip check reports the conflicts above"
