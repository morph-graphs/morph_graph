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
#   API_RUNTIME     docker (default) or apptainer (clusters without Docker: the same image with `apptainer exec`).
#                   MERIT_RUNTIME is read when API_RUNTIME is unset: it is what the job board's worker sets
#   API_IMAGE       image (default ghcr.io/morph-graphs/api:cpu-<env hash>, from image_tag.sh); for apptainer a
#                   .sif, or <MERIT_SIF_DIR>/api_cpu.sif when that exists, else the image is pulled (docker://)
#   API_HF_CACHE    Hugging Face cache on the host (default ~/.cache/huggingface)
#   API_LIMIT       qa: ask at most this many questions (a smoke test)
#   API_OUT, API_WORK   optional, paths inside the container (under /repo)
set -euo pipefail
: "${3:?usage: run.sh <stage> <video> <config> [bank]}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${API_IMAGE:-ghcr.io/morph-graphs/api:$(bash "$REPO/baselines/api/docker/image_tag.sh" cpu)}
if [ -z "${API_IMAGE:-}" ] && [ -n "${MERIT_SIF_DIR:-}" ] && [ -s "$MERIT_SIF_DIR/api_cpu.sif" ]; then
    IMAGE=$MERIT_SIF_DIR/api_cpu.sif
fi
RUNTIME=${API_RUNTIME:-${MERIT_RUNTIME:-docker}}
HF=${API_HF_CACHE:-$HOME/.cache/huggingface}
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
# a folder that is not a git checkout (a synced copy) records an empty commit instead of stopping the run
COMMITS=(API_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" API_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)")
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export APPTAINERENV_EPIC_MORPH_DIR=/epic-morph APPTAINERENV_HDEPIC_VIDEOS=/videos APPTAINERENV_HF_HOME=/hf
    export "${COMMITS[@]/#/APPTAINERENV_}"
    for v in GEMINI_API_KEY OPENAI_API_KEY HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX API_LIMIT API_OUT API_WORK; do
        [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"
    done
    exec apptainer exec --cleanenv --pwd /repo \
        --bind "$REPO:/repo,$EPIC:/epic-morph,$HF:/hf,$(cd -P "$VIDEOS" && pwd):/videos:ro" \
        "$IMAGE" bash baselines/api/run_api.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "API_RUNTIME must be docker or apptainer"; exit 1; }
exec docker run --rm --user "$(id -u):$(id -g)" \
    -v "$REPO:/repo" -v "$EPIC:/epic-morph" -v "$HF:/hf" -v "$(cd -P "$VIDEOS" && pwd):/videos:ro" -w /repo \
    -e EPIC_MORPH_DIR=/epic-morph -e HDEPIC_VIDEOS=/videos \
    -e "${COMMITS[0]}" -e "${COMMITS[1]}" \
    ${GEMINI_API_KEY:+-e GEMINI_API_KEY} ${OPENAI_API_KEY:+-e OPENAI_API_KEY} ${HF_TOKEN:+-e HF_TOKEN} \
    ${MERIT_HF_REPO:+-e MERIT_HF_REPO} ${MERIT_OUTBOX:+-e MERIT_OUTBOX} ${API_LIMIT:+-e API_LIMIT} \
    ${API_OUT:+-e API_OUT="$API_OUT"} ${API_WORK:+-e API_WORK="$API_WORK"} \
    "$IMAGE" bash baselines/api/run_api.sh "$@"
