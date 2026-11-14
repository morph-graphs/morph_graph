#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/egagent/docker/build.sh
# Tag: ghcr.io/morph-graphs/egagent:gpu-<environment hash> (image_tag.sh), the same tag
# .github/workflows/baseline-images.yml publishes; a local build stands in for a pull.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/egagent/docker/image_tag.sh" gpu)
docker build -f "$REPO/baselines/egagent/docker/Dockerfile" --build-arg ARM=gpu --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/egagent:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/egagent:$TAG (run.sh picks this tag for the checkout)"
