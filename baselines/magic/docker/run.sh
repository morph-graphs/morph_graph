#!/usr/bin/env bash
# Run run_magic.sh inside the arm's image: the morph_graph checkout is mounted at /repo (the code; results go
# to baselines/magic/results/ in it, then `upload` puts them in the Hugging Face dataset), the epic-morph
# checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or `hf auth login` on the host
# (the cache, with its token, is mounted). The same interface as baselines/merit/docker/run.sh, so the job
# board (coord/worker.py) runs both; it sets MERIT_GPU / MERIT_RUNTIME, which are read here as fallbacks.
#   MAGIC_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/magic/docker/run.sh <stage> <video> <arm> [bank]
#   MAGIC_RUNTIME   docker (default) or apptainer (clusters without Docker: the same image, run with
#                   `apptainer exec`; GPUs are the ones the scheduler gave the job, so MAGIC_GPU is not needed)
#   MAGIC_GPU       docker: GPU index(es) for this job, e.g. 0 or 2,3 (m8: one GPU >= 40 GB or two;
#                   m27: one 80 GB GPU or 3+)
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   MAGIC_IMAGE     image (default ghcr.io/morph-graphs/magic:<arm>-<env hash>, from image_tag.sh); for
#                   apptainer, a .sif built once with `apptainer pull magic_<arm>.sif docker://<that image>`
#   MAGIC_SIF_DIR   apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/magic_<arm>.sif when it
#                   exists and MAGIC_IMAGE is unset, so one board worker runs every baseline and arm
#   MERIT_WORK      optional, the cluster's shared host work folder (MERIT's), mounted at /work; MAGIC's work
#                   folder is then /work/magic (unless MAGIC_WORK is set), group-writable
#   MAGIC_HF_CACHE  Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface):
#                   Qwen/Qwen3-VL-8B-Instruct or Qwen/Qwen3.8-27B, Qwen/Qwen3-Embedding-4B, VLM2Vec/VLM2Vec-V2.0, Qwen/Qwen2-VL-2B-Instruct
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   MAGIC_OUT, MAGIC_WORK  optional, paths inside the container (e.g. /repo/baselines/magic/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO MAGIC_TAG MAGIC_FETCH TOKEN_BUDGET MAX_BATCH MAGIC_PORT
#   reported arms (oss120, oss120-flash, oss120-q27; README "Reported configuration"): OPENROUTER_API_KEY, the
#   MAGIC_{CAPTION,MEMORY,REASON}_{MODEL,BASE_URL,KEY_ENV} settings and the key variables those name, MAGIC_CAPTIONS,
#   MAGIC_CONTROLLER.
#   Keys go in through the environment like HF_TOKEN, never on a command line.
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMG=$(bash "$REPO/baselines/magic/docker/image_tag.sh" "$ARM"); IMG_ARM=${IMG%%-*}   # the reported arms use m8's or m27's image
IMAGE=${MAGIC_IMAGE:-ghcr.io/morph-graphs/magic:$IMG}
RUNTIME=${MAGIC_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${MAGIC_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${MAGIC_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/magic_$IMG_ARM.sif" ]; then
    IMAGE=$SIFS/magic_$IMG_ARM.sif
fi
HF=${MAGIC_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
for s in baselines/magic/MAGIC-video/src/worldmm baselines/merit/MERIT/eval; do   # MAGIC uses MERIT's prep and captions
    [ -d "$REPO/$s" ] || { echo "run: git submodule update --init baselines/magic/MAGIC-video baselines/merit/MERIT"; exit 1; }
done
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: MAGIC's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export MAGIC_WORK=${MAGIC_WORK:-/work/magic}
fi
# the container cannot run git on the checkout (a worktree's .git points outside /repo): pass the commits.
# MAGIC_REPO_COMMIT etc. may be set by the caller (a synced copy without .git).
rev() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
C_REPO=${MAGIC_REPO_COMMIT:-$(rev "$REPO")}; C_EPIC=${MAGIC_EPIC_COMMIT:-$(rev "$EPIC")}
COMMITS=(MAGIC_REPO_COMMIT="$C_REPO" MAGIC_EPIC_COMMIT="$C_EPIC"
         MAGIC_SUBMODULE_COMMIT="${MAGIC_SUBMODULE_COMMIT:-$(rev "$REPO/baselines/magic/MAGIC-video")}"
         MERIT_REPO_COMMIT="$C_REPO" MERIT_EPIC_COMMIT="$C_EPIC"
         MERIT_SUBMODULE_COMMIT="${MERIT_SUBMODULE_COMMIT:-$(rev "$REPO/baselines/merit/MERIT")}")
PASS=(HF_TOKEN MERIT_HF_REPO MAGIC_TAG MAGIC_FETCH MAGIC_OUT MAGIC_WORK TOKEN_BUDGET MAX_BATCH MAGIC_PORT
      OPENROUTER_API_KEY MAGIC_CAPTIONS MAGIC_CONTROLLER)
for r in CAPTION MEMORY REASON; do   # each role's settings, and the key variable its KEY_ENV names
    PASS+=(MAGIC_${r}_MODEL MAGIC_${r}_BASE_URL MAGIC_${r}_KEY_ENV); k=MAGIC_${r}_KEY_ENV
    [ -z "${!k:-}" ] || PASS+=("${!k}")
done
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`.
    # The host's network is shared, so SLURM_JOB_ID goes in too: run_magic.sh derives the server port from it.
    export APPTAINERENV_EPIC_MORPH_DIR=/epic-morph APPTAINERENV_HDEPIC_VIDEOS=/videos APPTAINERENV_HF_HOME=/hf
    export APPTAINERENV_HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} "${COMMITS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES SLURM_JOB_ID; do
        [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"
    done
    exec apptainer exec --nv --cleanenv --pwd /repo \
        --bind "$(IFS=,; echo "${BINDS[*]}")" "$IMAGE" bash baselines/magic/run_magic.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "MAGIC_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${MAGIC_GPU:-${MERIT_GPU:?set MAGIC_GPU to the GPU index or indices for this job}}
# the container sees only the chosen GPUs, renumbered from 0. Values go in through the environment (-e NAME
# with no value copies it from here), so the token never appears on a command line.
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); export CUDA_VISIBLE_DEVICES=$(seq 0 $((N - 1)) | paste -sd, -)
export EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} "${COMMITS[@]}"
ENVS=(-e CUDA_VISIBLE_DEVICES -e EPIC_MORPH_DIR -e HDEPIC_VIDEOS -e HF_HUB_OFFLINE)
for c in "${COMMITS[@]}"; do ENVS+=(-e "${c%%=*}"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ENVS+=(-e "$v"); done
for b in "${BINDS[@]}"; do ENVS+=(-v "$b"); done
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" \
    -w /repo "${ENVS[@]}" "$IMAGE" bash baselines/magic/run_magic.sh "$@"
