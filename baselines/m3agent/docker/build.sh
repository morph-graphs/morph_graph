#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/m3agent/docker/build.sh [m3]
# Tag: ghcr.io/morph-graphs/m3agent:m3-<environment hash> (image_tag.sh), the same tag
# .github/workflows/baseline-images.yml publishes; a local build stands in for a pull.
set -euo pipefail
ARM=${1:-m3}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/m3agent/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/m3agent/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/m3agent:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/m3agent:$TAG (run.sh picks this tag for the checkout)"
