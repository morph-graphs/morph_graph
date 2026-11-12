#!/usr/bin/env bash
# Install one arm's pinned packages into a Python that already has torch 2.6.0 (CUDA 12.4): first the
# prebuilt CUDA wheels in wheels-<arm>.txt (flash-attn), then the requirements (m27 also compiles causal-conv1d).
# Used by setup_env.sh (virtualenv) and docker/Dockerfile (image), so both get the same stack.
#   bash install_requirements.sh <m8|m27> <python>
# requirements-<arm>-nodeps.txt, when present, holds packages whose metadata forbids a version we pin
# but that work with it (checked in the batch10 runs); they are installed without their dependencies.
set -euo pipefail
ARM=${1:?m8 or m27}; P=${2:?python}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$P" -m pip install -q --upgrade pip wheel setuptools packaging ninja   # build tools for --no-build-isolation
"$P" -m pip install -q --no-build-isolation $(grep -v '^#' "$HERE/wheels-$ARM.txt")
"$P" -m pip install -q --no-build-isolation -r "$HERE/requirements-$ARM.txt"
if [ "$ARM" = m27 ]; then
    # causal-conv1d 1.6.1 (as in the batch10 27B runs), compiled: its prebuilt torch2.6 wheels do not load with
    # torch 2.6.0. Needs nvcc (CUDA 12.4, e.g. CUDA_HOME=/usr/local/cuda in the devel image); ~5-10 min.
    # FORCE_BUILD stops setup.py from fetching that broken wheel; the arch list covers A100, A40/A5000/A6000,
    # L40S and H100.
    CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}" MAX_JOBS="${MAX_JOBS:-8}" \
        "$P" -m pip install -q --no-build-isolation causal-conv1d==1.6.1
    "$P" -c "import causal_conv1d" || { echo "causal-conv1d does not import (needs nvcc and CUDA_HOME)"; exit 1; }
fi
if [ -s "$HERE/requirements-$ARM-nodeps.txt" ]; then
    "$P" -m pip install -q --no-deps -r "$HERE/requirements-$ARM-nodeps.txt"
fi
"$P" -m pip check || echo "pip check reports the conflicts above (expected only for requirements-$ARM-nodeps.txt)"
