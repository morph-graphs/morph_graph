#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/rekv/docker/build.sh m8
# Tag: ghcr.io/morph-graphs/rekv:<arm>-<environment hash> (image_tag.sh); a local build stands in for a pull.
set -euo pipefail
ARM=${1:?m8}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/rekv/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/rekv/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/rekv:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/rekv:$TAG (run.sh picks this tag for the checkout)"
