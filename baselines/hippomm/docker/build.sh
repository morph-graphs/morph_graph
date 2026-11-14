#!/usr/bin/env bash
# Build the arm's image from the repo root's current commit: bash baselines/hippomm/docker/build.sh q25vl7b-gpt4o
# Tag: ghcr.io/morph-graphs/hippomm:<arm>-<environment hash> (image_tag.sh); a local build stands in for a pull.
set -euo pipefail
ARM=${1:?q25vl7b-gpt4o}; REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); C=$(git -C "$REPO" rev-parse --short HEAD)
TAG=$(bash "$REPO/baselines/hippomm/docker/image_tag.sh" "$ARM")
docker build -f "$REPO/baselines/hippomm/docker/Dockerfile" --build-arg ARM="$ARM" --build-arg COMMIT="$C" \
    -t "ghcr.io/morph-graphs/hippomm:$TAG" "$REPO"
echo "built ghcr.io/morph-graphs/hippomm:$TAG (run.sh picks this tag for the checkout)"
