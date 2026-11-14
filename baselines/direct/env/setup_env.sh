#!/usr/bin/env bash
# The direct baseline runs in the MERIT environment of its model (same loader, same pinned packages), so
# there is nothing of its own to pin: this builds that environment.
#   bash baselines/direct/env/setup_env.sh <model> <venv dir>
#   m8 (transformers 4.57.3): qwen3vl-8b, molmo2-8b     m27 (transformers 5.8.0): qwen38-27b, qwen35-9b, qwen35-35b-a3b
set -euo pipefail
MODEL=${1:?model, e.g. qwen3vl-8b}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $MODEL in qwen3vl-8b|molmo2-8b) ARM=m8 ;; qwen38-27b|qwen35-9b|qwen35-35b-a3b) ARM=m27 ;; *) echo "unknown model $MODEL"; exit 1 ;; esac
exec bash "$HERE/../../merit/env/setup_env.sh" "$ARM" "$DIR"
