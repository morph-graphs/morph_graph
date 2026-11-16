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
#   passed on when set: OPENAI_BASE_URL (a gateway), OPENROUTER_API_KEY (configs that call OpenRouter),
#                   EGAGENT_GPT_PREFIX (a gateway's prefix to GPT model ids, configs with name_prefix_env), EVCLIPS_HF_REPO
#   EVCLIPS_DIR     optional, a local copy of the evidence-clips videos (the *-evclips configs; else they are fetched
#                   from the dataset), mounted read-only at /evclips
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
BINDS="$REPO:/repo,$EPIC:/epic-morph,$HF:/hf,$(cd -P "$VIDEOS" && pwd):/videos:ro"
EVC=(); if [ -n "${EVCLIPS_DIR:-}" ]; then BINDS+=",$(cd -P "$EVCLIPS_DIR" && pwd):/evclips:ro"; EVC=(-e EVCLIPS_DIR=/evclips); fi
PASS=(GEMINI_API_KEY OPENAI_API_KEY OPENAI_BASE_URL OPENROUTER_API_KEY EGAGENT_GPT_PREFIX EVCLIPS_HF_REPO HF_TOKEN MERIT_HF_REPO
      MERIT_OUTBOX API_LIMIT API_OUT API_WORK)
COMMITS=(API_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" API_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)")
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export APPTAINERENV_EPIC_MORPH_DIR=/epic-morph APPTAINERENV_HDEPIC_VIDEOS=/videos APPTAINERENV_HF_HOME=/hf
    export "${COMMITS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}"; do
        [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"
    done
    [ -z "${EVCLIPS_DIR:-}" ] || export APPTAINERENV_EVCLIPS_DIR=/evclips
    exec apptainer exec --cleanenv --pwd /repo --bind "$BINDS" "$IMAGE" bash baselines/api/run_api.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "API_RUNTIME must be docker or apptainer"; exit 1; }
ARGS=(); IFS=, read -ra B <<< "$BINDS"; for b in "${B[@]}"; do ARGS+=(-v "$b"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
exec docker run --rm --user "$(id -u):$(id -g)" "${ARGS[@]}" -w /repo -e EPIC_MORPH_DIR=/epic-morph -e HDEPIC_VIDEOS=/videos \
    -e HF_HOME=/hf -e "${COMMITS[0]}" -e "${COMMITS[1]}" ${EVC[@]+"${EVC[@]}"} "$IMAGE" bash baselines/api/run_api.sh "$@"
