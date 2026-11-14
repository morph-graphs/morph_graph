#!/usr/bin/env bash
# The image tag: gemini25pro-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit on main needs no rebuild. One image serves
# every API arm (it holds the environment, not the model's name; it keeps the name of the first arm); an arm
# name is accepted so callers can pass theirs.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml.
#   bash baselines/egor1/docker/image_tag.sh [gemini25pro|gemini36flash]
set -euo pipefail
ARM=${1:-gemini25pro}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $ARM in gemini25pro|gemini36flash) ;; *) echo "the arms with an image are gemini25pro and gemini36flash (the same one)" >&2; exit 1 ;; esac
H=$(cd "$M" && cat docker/Dockerfile env/requirements.txt | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "gemini25pro-$H"
