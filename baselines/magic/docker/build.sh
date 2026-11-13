#!/usr/bin/env bash
# Build one arm's image from the repo root's current commit: bash baselines/magic/docker/build.sh <m8|m27>
# Tag: ghcr.io/morph-graphs/magic:<arm>-<environment hash> (image_tag.sh), the same tag
# .github/workflows/baseline-images.yml publishes; a local build stands in for a pull.
set -euo pipefail
ARM=${1:?m8 or m27}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/magic/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/magic/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/magic:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/magic:$TAG (run.sh picks this tag for the checkout)"
