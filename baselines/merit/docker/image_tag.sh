#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with
# the same environment files maps to the same image, so a code-only commit on main needs no rebuild.
# Used by build.sh, run.sh and .github/workflows/merit-images.yml.
#   bash baselines/merit/docker/image_tag.sh <m8|m27>
set -euo pipefail
ARM=${1:?m8 or m27}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/wheels-"$ARM".txt env/requirements-"$ARM".txt \
    $(ls env/requirements-"$ARM"-nodeps.txt 2>/dev/null) | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
