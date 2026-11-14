#!/usr/bin/env bash
# The image tag: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit on main needs no rebuild.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml.
#   bash baselines/egorag/docker/image_tag.sh [egogpt_gpt4o]
set -euo pipefail
ARM=${1:-egogpt_gpt4o}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ "$ARM" = egogpt_gpt4o ] || { echo "the only arm is egogpt_gpt4o" >&2; exit 1; }
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/requirements-nobuildiso.txt env/requirements.txt \
    | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
