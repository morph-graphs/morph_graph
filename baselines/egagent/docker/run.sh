#!/usr/bin/env bash
# Run run_egagent.sh inside the image (as MERIT's and WorldMM's docker/run.sh): the morph_graph checkout is
# mounted at /repo (the code; results go to baselines/egagent/results/ in it, then `upload` puts them in the
# Hugging Face dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos.
#   EPIC_MORPH_DIR=<epic-morph checkout> OPENAI_API_KEY=... bash baselines/egagent/docker/run.sh <stage> <video> <arm> [bank]
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   OPENAI_API_KEY   for build (the memory, both arms) and for qa / blind of arm gpt41
#   GEMINI_API_KEY   for qa / blind of arm gemini25pro
#                    Export a key in the shell: it is handed to the container by name, so it never appears on a
#                    command line, in a file or in the image. A paid stage without its key stops, nothing spent
#   HF_TOKEN         for upload and for taking a memory from the dataset; or `hf auth login` on the host (the
#                    cache, with its token, is mounted)
#   EGAGENT_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image with `apptainer exec`)
#   EGAGENT_GPU      docker: GPU index for this job, e.g. 0 (the SigLIP 2 embedder; about 8 GB). Unset: CPU,
#                    fine for qa and blind, slow for build
#   EGAGENT_IMAGE    image (default ghcr.io/morph-graphs/egagent:gpu-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull egagent_gpu.sif docker://<that image>`
#   EGAGENT_SIF_DIR  apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/egagent_gpu.sif when
#                    it exists and EGAGENT_IMAGE is unset
#   EGAGENT_HF_CACHE Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   MERIT_WORK       optional, the cluster's shared host work folder (MERIT's), mounted at /work; EGAgent's work
#                    folder is then /work/egagent (unless EGAGENT_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the EGAGENT_ ones are unset.
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   EGAGENT_OUT, EGAGENT_WORK  optional, paths inside the container (e.g. /repo/baselines/egagent/work/check)
#   passed through when set: MERIT_HF_REPO EGAGENT_TAG EGAGENT_FETCH EGAGENT_FRAME_PX EGAGENT_CHUNK_S THREADS LIMIT
set -euo pipefail
: "${3:?usage: run.sh <stage> <video> <arm> [bank]}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${EGAGENT_IMAGE:-ghcr.io/morph-graphs/egagent:$(bash "$REPO/baselines/egagent/docker/image_tag.sh" gpu)}
RUNTIME=${EGAGENT_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${EGAGENT_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${EGAGENT_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/egagent_gpu.sif" ]; then
    IMAGE=$SIFS/egagent_gpu.sif
fi
HF=${EGAGENT_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/egagent/egagent/egagent/langgraph_agent.py" ] || { echo "run: git submodule update --init baselines/egagent/egagent"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
# a folder that is not a git checkout (a synced copy) records an empty commit instead of stopping the run
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      EGAGENT_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
      EGAGENT_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)"
      EGAGENT_UPSTREAM_COMMIT="$(git -C "$REPO/baselines/egagent/egagent" rev-parse HEAD 2>/dev/null || true)")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: EGAgent's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export EGAGENT_WORK=${EGAGENT_WORK:-/work/egagent}
fi
PASS=(OPENAI_API_KEY GEMINI_API_KEY HF_TOKEN MERIT_HF_REPO EGAGENT_TAG EGAGENT_FETCH EGAGENT_FRAME_PX EGAGENT_CHUNK_S
      THREADS LIMIT EGAGENT_OUT EGAGENT_WORK)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/egagent/run_egagent.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "EGAGENT_RUNTIME must be docker or apptainer"; exit 1; }
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
GPU=${EGAGENT_GPU:-${MERIT_GPU:-}}
if [ -n "$GPU" ]; then   # the container sees only the chosen GPU(s), renumbered from 0
    N=$(echo "$GPU" | tr ',' '\n' | grep -c .); ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N - 1)))")
fi
exec docker run --rm --shm-size 8g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" bash baselines/egagent/run_egagent.sh "$@"
