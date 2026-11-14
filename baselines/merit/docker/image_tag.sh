#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with
# the same environment files maps to the same image, so a code-only commit on main needs no rebuild.
# Used by build.sh, run.sh and .github/workflows/baseline-images.yml. IMAGE_ARCH=arm64 gives the aarch64 image's tag,
# <arm>-arm64-<hash> over Dockerfile.arm64 and its files (coord/worker.py sets it on aarch64 hosts).
#   bash baselines/merit/docker/image_tag.sh <m8|m27>
set -euo pipefail
ARM=${1:?m8 or m27}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=(docker/Dockerfile env/wheels-"$ARM".txt); A=
[ "${IMAGE_ARCH:-}" = arm64 ] && { D=(docker/Dockerfile.arm64 env/wheels-"$ARM"-arm64.txt ../common/docker/arm64_stack.sh); A=arm64-; }
H=$(cd "$M" && cat "${D[0]}" env/install_requirements.sh "${D[@]:1}" env/requirements-"$ARM".txt \
    $(ls env/requirements-"$ARM"-nodeps.txt 2>/dev/null) | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$A$H"
