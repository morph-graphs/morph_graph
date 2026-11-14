#!/usr/bin/env bash
# Build the image from the repo root's current commit: bash baselines/egorag/docker/build.sh [egogpt_gpt4o]
# Tag: ghcr.io/morph-graphs/egorag:<arm>-<environment hash> (image_tag.sh); a local build stands in for a pull.
set -euo pipefail
ARM=${1:-egogpt_gpt4o}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/egorag/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/egorag/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/egorag:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/egorag:$TAG (run.sh picks this tag for the checkout)"
