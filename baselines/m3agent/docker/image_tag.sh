#!/usr/bin/env bash
# The image tag: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit on main needs no rebuild. As MERIT's.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml.
#   bash baselines/m3agent/docker/image_tag.sh [m3]
set -euo pipefail
ARM=${1:-m3}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ "$ARM" = m3 ] || { echo "the only arm is m3" >&2; exit 1; }
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/wheels-mem.txt env/requirements-mem.txt \
    env/requirements-ctl.txt | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
