#!/usr/bin/env bash
# Run run_worldmm.sh inside the arm's image (as MERIT's docker/run.sh): the morph_graph checkout is mounted at
# /repo (the code; results go to baselines/worldmm/results/ in it, then `upload` puts them in the Hugging Face
# dataset), the epic-morph checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or
# `hf auth login` on the host (the cache, with its token, is mounted).
#   WORLDMM_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/worldmm/docker/run.sh <stage> <video> <arm> [bank]
#   WORLDMM_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image, run with
#                    `apptainer exec`; GPUs are the ones the scheduler gave the job, so WORLDMM_GPU is not needed)
#   WORLDMM_GPU      docker: GPU index(es) for this job, e.g. 0 (m8) or 2,3 (m27 on 2x 48 GB)
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   WORLDMM_IMAGE    image (default ghcr.io/morph-graphs/worldmm:<arm>-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull worldmm_<arm>.sif docker://<that image>`
#   WORLDMM_SIF_DIR  apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/worldmm_<arm>.sif (the official arms: worldmm_m8.sif) when
#                    it exists and WORLDMM_IMAGE is unset, so one board worker runs every baseline and arm
#   WORLDMM_HF_CACHE Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   MERIT_WORK       optional, the cluster's shared host work folder (MERIT's), mounted at /work; WorldMM's work
#                    folder is then /work/worldmm (unless WORLDMM_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the WORLDMM_ ones are unset.
#   WORLDMM_MODEL_DIR optional, a host folder of model folders, mounted read-only at /models
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   WORLDMM_OUT, WORLDMM_WORK  optional, paths inside the container (e.g. /repo/baselines/worldmm/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO WORLDMM_TAG WORLDMM_FETCH WORLDMM_EMB_BATCH
#                    WORLDMM_MAX_MEMORY WORLDMM_SERVER_WAIT MAX_BATCH THREADS WORLDMM_OPENAI_SERVICE_TIER, and for the
#                    official arms only OPENAI_API_KEY (by name through the environment, never on a command line) and
#                    OPENAI_BASE_URL (another endpoint: test_official.sh's stand-in); OPENROUTER_API_KEY without
#                    OPENAI_API_KEY sends the same calls through OpenRouter
# The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME for every baseline: read here as fallbacks.
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${WORLDMM_IMAGE:-ghcr.io/morph-graphs/worldmm:$(bash "$REPO/baselines/worldmm/docker/image_tag.sh" "$ARM")}
RUNTIME=${WORLDMM_RUNTIME:-${MERIT_RUNTIME:-docker}}
case $ARM in g5m*) ;; *) unset OPENAI_API_KEY OPENROUTER_API_KEY ;; esac   # only the official arms' containers get the key
SIFS=${WORLDMM_SIF_DIR:-${MERIT_SIF_DIR:-}}
case $ARM in g5m*) ENV=m8 ;; *) ENV=$ARM ;; esac        # the image (environment) the arm runs in, as image_tag.sh
if [ "$RUNTIME" = apptainer ] && [ -z "${WORLDMM_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/worldmm_$ENV.sif" ]; then
    IMAGE=$SIFS/worldmm_$ENV.sif
fi
HF=${WORLDMM_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/worldmm/WorldMM/src/worldmm/memory/memory.py" ] || { echo "run: git submodule update --init baselines/worldmm/WorldMM"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      WORLDMM_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)" WORLDMM_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD)"
      WORLDMM_SUBMODULE_COMMIT="$(git -C "$REPO/baselines/worldmm/WorldMM" rev-parse HEAD)")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: WorldMM's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export WORLDMM_WORK=${WORLDMM_WORK:-/work/worldmm}
fi
if [ -n "${WORLDMM_MODEL_DIR:-}" ]; then BINDS+=("$(cd -P "$WORLDMM_MODEL_DIR" && pwd):/models:ro"); ENVS+=(WORLDMM_MODEL_DIR=/models); fi
PASS=(HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX WORLDMM_TAG WORLDMM_FETCH WORLDMM_EMB_BATCH WORLDMM_MAX_MEMORY WORLDMM_SERVER_WAIT
      MAX_BATCH THREADS WORLDMM_OUT WORLDMM_WORK OPENAI_API_KEY OPENAI_BASE_URL OPENROUTER_API_KEY OPENROUTER_BASE_URL WORLDMM_OPENAI_SERVICE_TIER)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    # apptainer shares the host's /tmp: a per-user Triton cache, not the image's /tmp/triton
    export APPTAINERENV_TRITON_CACHE_DIR=/tmp/triton_$(id -u)
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/worldmm/run_worldmm.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "WORLDMM_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${WORLDMM_GPU:-${MERIT_GPU:?set WORLDMM_GPU to the GPU index or indices for this job}}
# the container sees only the chosen GPUs, renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" -w /repo \
    -e CUDA_VISIBLE_DEVICES="$SEEN" "${ARGS[@]}" "$IMAGE" bash baselines/worldmm/run_worldmm.sh "$@"
