#!/usr/bin/env bash
# Run run_rekv.sh inside the arm's image (as MERIT's docker/run.sh): the morph_graph checkout is mounted at
# /repo (the code; results go to baselines/rekv/results/ in it, then `upload` puts them in the Hugging Face
# dataset), the epic-morph checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or
# `hf auth login` on the host (the cache, with its token, is mounted).
#   REKV_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/rekv/docker/run.sh <stage> <video> <arm> [bank]
#   REKV_RUNTIME    docker (default) or apptainer (clusters without Docker: the same image, run with
#                   `apptainer exec`; GPUs are the ones the scheduler gave the job, so REKV_GPU is not needed)
#   REKV_GPU        docker: GPU index(es) for this job: one 48 GB GPU, e.g. 0, or two 24 GB, e.g. 0,1
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   REKV_IMAGE      image (default ghcr.io/morph-graphs/rekv:<arm>-<env hash>, from image_tag.sh); for apptainer,
#                   a .sif built once with `apptainer pull rekv_<arm>.sif docker://<that image>`
#   REKV_SIF_DIR    apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/rekv_<arm>.sif when it
#                   exists and REKV_IMAGE is unset, so one board worker runs every baseline and arm
#   REKV_HF_CACHE   Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   MERIT_WORK      optional, the cluster's shared host work folder (MERIT's), mounted at /work. ReKV keeps
#                   nothing between runs (the KV-cache lives in RAM), so only Triton's cache goes there
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the REKV_ ones are unset.
#   REKV_MODEL_DIR  optional, a host folder with a local copy of the model (Qwen3-VL-8B-Instruct), mounted
#                   read-only at /models
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   REKV_OUT        optional, a path inside the container (e.g. /repo/baselines/rekv/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX REKV_TAG REKV_LIMIT
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${REKV_IMAGE:-ghcr.io/morph-graphs/rekv:$(bash "$REPO/baselines/rekv/docker/image_tag.sh" "$ARM")}
RUNTIME=${REKV_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${REKV_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${REKV_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/rekv_$ARM.sif" ]; then
    IMAGE=$SIFS/rekv_$ARM.sif
fi
HF=${REKV_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      REKV_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)" REKV_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD)")
TRITON=/tmp/triton_$(id -u)   # apptainer shares the host's /tmp: a per-user Triton cache
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: Triton's cache in ReKV's own subfolder
    umask 002; mkdir -p "$MERIT_WORK/rekv"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); TRITON=/work/rekv/triton_$(id -u)
fi
if [ -n "${REKV_MODEL_DIR:-}" ]; then BINDS+=("$(cd -P "$REKV_MODEL_DIR" && pwd):/models:ro"); ENVS+=(REKV_MODELS=/models); fi
PASS=(HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX REKV_TAG REKV_LIMIT REKV_OUT)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    export APPTAINERENV_TRITON_CACHE_DIR=$TRITON
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/rekv/run_rekv.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "REKV_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${REKV_GPU:-${MERIT_GPU:?set REKV_GPU to the GPU index or indices for this job}}
# the container sees only the chosen GPUs, renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" -w /repo \
    -e CUDA_VISIBLE_DEVICES="$SEEN" "${ARGS[@]}" "$IMAGE" bash baselines/rekv/run_rekv.sh "$@"
