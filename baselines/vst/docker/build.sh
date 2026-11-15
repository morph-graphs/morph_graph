#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/vst/docker/build.sh vst7b
# Tag: ghcr.io/morph-graphs/vst:<arm>-<environment hash> (image_tag.sh); a local build stands in for a pull.
set -euo pipefail
ARM=${1:?vst7b}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/vst/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/vst/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/vst:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/vst:$TAG (run.sh picks this tag for the checkout)"
