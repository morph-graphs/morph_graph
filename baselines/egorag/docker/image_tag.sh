#!/usr/bin/env bash
# The image tag: egogpt_gpt4o-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit on main needs no rebuild. One image serves every
# arm (the arms differ in the GPT model they ask, not in the environment); it carries the first arm's name.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml.
#   bash baselines/egorag/docker/image_tag.sh [egogpt_gpt4o|egogpt_gpt41]
set -euo pipefail
ARM=${1:-egogpt_gpt4o}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $ARM in egogpt_gpt4o|egogpt_gpt41) ;; *) echo "the arms are egogpt_gpt4o and egogpt_gpt41" >&2; exit 1 ;; esac
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/requirements-nobuildiso.txt env/requirements.txt \
    | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "egogpt_gpt4o-$H"
