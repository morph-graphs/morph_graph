#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the
# same environment files maps to the same image, so a code-only commit needs no rebuild. As MERIT's.
#   bash baselines/vst/docker/image_tag.sh vst7b        (vst7b-evclips, the evidence-clips arm: the same image)
set -euo pipefail
ARM=${1:?vst7b}; ARM=${ARM%-evclips}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ "$ARM" = vst7b ] || { echo "arm must be vst7b" >&2; exit 1; }
# x86-64 only: the pinned flash-attn wheel is linux_x86_64, and there is no Dockerfile.arm64 yet
[ "${IMAGE_ARCH:-}" != arm64 ] || { echo "VST has no arm64 image (baselines/vst/docker has no Dockerfile.arm64)" >&2; exit 1; }
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/wheels-"$ARM".txt env/requirements-"$ARM".txt \
    | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
