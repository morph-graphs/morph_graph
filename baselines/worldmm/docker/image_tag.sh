#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the
# same environment files maps to the same image, so a code-only commit needs no rebuild. As MERIT's, IMAGE_ARCH=arm64
# included (the aarch64 image's tag, <arm>-arm64-<hash>).
#   bash baselines/worldmm/docker/image_tag.sh <m8|m27|g5m|g5m-q8|g5m-g5>   (the official arms run in m8's image)
set -euo pipefail
ARM=${1:?m8 or m27}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $ARM in
    m8|g5m|g5m-q8|g5m-g5|g5m-g51) ARM=m8; ENVS=(client) ;;
    m27) ENVS=(client server27) ;;
    *) echo "arm must be m8, m27, g5m, g5m-q8, g5m-g5 or g5m-g51" >&2; exit 1 ;;
esac
X=; A=; [ "${IMAGE_ARCH:-}" = arm64 ] && { X=-arm64; A=arm64-; }
FILES=(docker/Dockerfile${X/-/.} env/install_requirements.sh env/setup_env.sh)
for e in "${ENVS[@]}"; do
    FILES+=(env/wheels-"$e"$X.txt env/requirements-"$e".txt)
    [ -f "$M/env/requirements-$e-nodeps.txt" ] && FILES+=(env/requirements-"$e"-nodeps.txt)
done
[ -n "$X" ] && FILES+=(../common/docker/arm64_stack.sh)
H=$(cd "$M" && cat "${FILES[@]}" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$A$H"
