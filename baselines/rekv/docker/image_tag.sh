#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the
# same environment files maps to the same image, so a code-only commit needs no rebuild. As MERIT's.
#   bash baselines/rekv/docker/image_tag.sh m8
set -euo pipefail
ARM=${1:?m8}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ "$ARM" = m8 ] || { echo "arm must be m8" >&2; exit 1; }
H=$(cd "$M" && cat docker/Dockerfile env/requirements.txt env/setup_env.sh | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
