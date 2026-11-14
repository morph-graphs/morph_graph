#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/egor1/docker/build.sh
# Tag: ghcr.io/morph-graphs/egor1:gemini25pro-<environment hash> (image_tag.sh), the same tag
# .github/workflows/baseline-images.yml publishes; a local build stands in for a pull.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/egor1/docker/image_tag.sh" gemini25pro)
docker build -f "$REPO/baselines/egor1/docker/Dockerfile" --build-arg ARM=gemini25pro --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/egor1:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/egor1:$TAG (run.sh picks this tag for the checkout)"
