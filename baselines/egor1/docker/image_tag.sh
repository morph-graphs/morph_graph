#!/usr/bin/env bash
# The image tag: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit on main needs no rebuild.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml.
#   bash baselines/egor1/docker/image_tag.sh [gemini25pro]
set -euo pipefail
ARM=${1:-gemini25pro}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ "$ARM" = gemini25pro ] || { echo "the only arm with an image is gemini25pro" >&2; exit 1; }
H=$(cd "$M" && cat docker/Dockerfile env/requirements.txt | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
