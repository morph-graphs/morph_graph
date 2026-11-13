#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/api/docker/build.sh
# Tag: ghcr.io/morph-graphs/api:cpu-<environment hash> (image_tag.sh), the same tag
# .github/workflows/baseline-images.yml publishes; a local build stands in for a pull.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/api/docker/image_tag.sh" cpu)
docker build -f "$REPO/baselines/api/docker/Dockerfile" --build-arg ARM=cpu --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/api:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/api:$TAG (run.sh picks this tag for the checkout)"
