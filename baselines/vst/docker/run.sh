#!/usr/bin/env bash
# Run run_vst.sh inside the arm's image (as MERIT's and ReKV's docker/run.sh): the morph_graph checkout is mounted
# at /repo (the code and the VST submodule; results go to baselines/vst/results/ in it, then `upload` puts them in
# the Hugging Face dataset), the epic-morph checkout at /epic-morph (the banks). The Hugging Face token (for
# `upload` only; the model is a public download): HF_TOKEN, or `hf auth login` on the host (the cache, with its
# token, is mounted).
#   VST_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/vst/docker/run.sh <stage> <video> <arm> [bank]
#   VST_RUNTIME     docker (default) or apptainer (clusters without Docker: the same image, run with
#                   `apptainer exec`; the GPU is the one the scheduler gave the job, so VST_GPU is not needed)
#   VST_GPU         docker: the GPU index for this job: one GPU with >= 24 GB (the model is loaded on one GPU)
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   VST_IMAGE       image (default ghcr.io/morph-graphs/vst:<arm>-<env hash>, from image_tag.sh); for apptainer,
#                   a .sif built once with `apptainer pull vst_<arm>.sif docker://<that image>`
#   VST_SIF_DIR     apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/vst_<arm>.sif when it
#                   exists and VST_IMAGE is unset, so one board worker runs every baseline and arm
#   VST_HF_CACHE    Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the VST_ ones are unset.
#   VST_MODEL_DIR   optional, a host folder with a local copy of the model (VST-7B), mounted read-only at /models
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   VST_OUT         optional, a path inside the container (e.g. /repo/baselines/vst/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX VST_TAG VST_LIMIT VST_THINK_TIMES VST_VID_TOKENS
#                   VST_MAX_FRAMES
# Host RAM: 64 GB for the job (the loader holds the sampled source frames as floats while it resizes them).
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
TAG=$(bash "$REPO/baselines/vst/docker/image_tag.sh" "$ARM") || exit 1   # stops here on an aarch64 host: no arm64 image
IMAGE=${VST_IMAGE:-ghcr.io/morph-graphs/vst:$TAG}
RUNTIME=${VST_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${VST_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${VST_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/vst_${ARM%-evclips}.sif" ]; then
    IMAGE=$SIFS/vst_${ARM%-evclips}.sif
fi
[ -f "$REPO/baselines/vst/VST/eval/lmms-eval/lmms_eval/models/simple/qwen2_5_vl_stream_think.py" ] || { echo "run: git submodule update --init baselines/vst/VST"; exit 1; }
HF=${VST_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      # || true: a folder without .git (a synced copy) gives an empty commit, not the end of the job
      VST_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
      VST_SUBMODULE_COMMIT="$(git -C "$REPO/baselines/vst/VST" rev-parse HEAD 2>/dev/null || true)"
      VST_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)")
if [ -n "${VST_MODEL_DIR:-}" ]; then BINDS+=("$(cd -P "$VST_MODEL_DIR" && pwd):/models:ro"); ENVS+=(VST_MODELS=/models); fi
PASS=(HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX VST_TAG VST_LIMIT VST_OUT VST_THINK_TIMES VST_VID_TOKENS VST_MAX_FRAMES EVCLIPS_HF_REPO)
# the evidence-clips arms' stitched videos (baselines/common/evclips.py): a local copy when EVCLIPS_DIR is set, else
# they are fetched from the dataset into the mounted cache
if [ -n "${EVCLIPS_DIR:-}" ]; then BINDS+=("$(cd -P "$EVCLIPS_DIR" && pwd):/evclips:ro"); ENVS+=(EVCLIPS_DIR=/evclips); fi
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/vst/run_vst.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "VST_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${VST_GPU:-${MERIT_GPU:?set VST_GPU to the GPU index for this job}}
# the container sees only the chosen GPU(s), renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" -w /repo \
    -e CUDA_VISIBLE_DEVICES="$SEEN" "${ARGS[@]}" "$IMAGE" bash baselines/vst/run_vst.sh "$@"
