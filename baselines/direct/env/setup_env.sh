#!/usr/bin/env bash
# The direct baseline runs in the MERIT environment of its model (same loader, same pinned packages), so
# there is nothing of its own to pin: this builds that environment.
#   bash baselines/direct/env/setup_env.sh <qwen3vl-8b|qwen38-27b> <venv dir>
set -euo pipefail
MODEL=${1:?qwen3vl-8b or qwen38-27b}; DIR=${2:?venv dir}; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $MODEL in qwen3vl-8b) ARM=m8 ;; qwen38-27b) ARM=m27 ;; *) echo "model must be qwen3vl-8b or qwen38-27b"; exit 1 ;; esac
exec bash "$HERE/../../merit/env/setup_env.sh" "$ARM" "$DIR"
