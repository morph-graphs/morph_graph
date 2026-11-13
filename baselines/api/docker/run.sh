#!/usr/bin/env bash
# Run run_api.sh inside the image: the morph_graph checkout is mounted at /repo (the code; clips and run folders
# go to baselines/api/work/ in it, results to baselines/api/results/, then `upload` puts them in the Hugging
# Face dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos.
#   EPIC_MORPH_DIR=<epic-morph checkout> GEMINI_API_KEY=... bash baselines/api/docker/run.sh <stage> <video> <config> [bank]
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   GEMINI_API_KEY / OPENAI_API_KEY   the provider's key, for qa. Export it in the shell: it is handed to the
#                   container by name, so it never appears on a command line, in a file or in the image
#   HF_TOKEN        for upload; or `hf auth login` on the host (the cache, with its token, is mounted)
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos)
#   API_IMAGE       image (default ghcr.io/morph-graphs/api:cpu-<env hash>, from image_tag.sh)
#   API_HF_CACHE    Hugging Face cache on the host (default ~/.cache/huggingface)
#   API_LIMIT       qa: ask at most this many questions (a smoke test)
#   API_OUT, API_WORK   optional, paths inside the container (under /repo)
set -euo pipefail
: "${3:?usage: run.sh <stage> <video> <config> [bank]}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${API_IMAGE:-ghcr.io/morph-graphs/api:$(bash "$REPO/baselines/api/docker/image_tag.sh" cpu)}
HF=${API_HF_CACHE:-$HOME/.cache/huggingface}
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
exec docker run --rm --user "$(id -u):$(id -g)" \
    -v "$REPO:/repo" -v "$EPIC:/epic-morph" -v "$HF:/hf" -v "$(cd -P "$VIDEOS" && pwd):/videos:ro" -w /repo \
    -e EPIC_MORPH_DIR=/epic-morph -e HDEPIC_VIDEOS=/videos \
    -e API_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)" -e API_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD)" \
    ${GEMINI_API_KEY:+-e GEMINI_API_KEY} ${OPENAI_API_KEY:+-e OPENAI_API_KEY} ${HF_TOKEN:+-e HF_TOKEN} \
    ${MERIT_HF_REPO:+-e MERIT_HF_REPO} ${API_LIMIT:+-e API_LIMIT} \
    ${API_OUT:+-e API_OUT="$API_OUT"} ${API_WORK:+-e API_WORK="$API_WORK"} \
    "$IMAGE" bash baselines/api/run_api.sh "$@"
