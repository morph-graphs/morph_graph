#!/usr/bin/env bash
# Run run_hippomm.sh inside the image (as MERIT's docker/run.sh): the morph_graph checkout is mounted at /repo
# (the code; results go to baselines/hippomm/results/ in it, then `upload` puts them in the Hugging Face
# dataset), the epic-morph checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or
# `hf auth login` on the host (the cache, with its token, is mounted).
#   HIPPOMM_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/hippomm/docker/run.sh <stage> <video> <arm> [bank]
#   OPENAI_API_KEY   for qa and blind (gpt-4o, upstream's reasoning model). Export it in the shell: it is handed
#                    to the container by name, so it never appears on a command line, in a file or in the image
#   HIPPOMM_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image, run with
#                    `apptainer exec`; the GPU is the one the scheduler gave the job, so HIPPOMM_GPU is not needed)
#   HIPPOMM_GPU      docker: the GPU index for this job (one GPU with 48 GB)
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   HIPPOMM_IMAGE    image (default ghcr.io/morph-graphs/hippomm:<arm>-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull hippomm_<arm>.sif docker://<that image>`
#   HIPPOMM_SIF_DIR  apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/hippomm_<arm>.sif when
#                    it exists and HIPPOMM_IMAGE is unset, so one board worker runs every baseline and arm
#   HIPPOMM_HF_CACHE Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted
#                    at /hf: Qwen2.5-VL-7B-Instruct, the Whisper model, and imagebind/imagebind_huge.pth (4.5 GB,
#                    downloaded there on first use) and the tokenizer file of gpt-4o
#   MERIT_WORK       optional, the cluster's shared host work folder (MERIT's), mounted at /work; HippoMM's work
#                    folder is then /work/hippomm (unless HIPPOMM_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the HIPPOMM_ ones are unset.
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   HIPPOMM_OUT, HIPPOMM_WORK  optional, paths inside the container (e.g. /repo/baselines/hippomm/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO HIPPOMM_TAG HIPPOMM_FETCH HIPPOMM_LIMIT HIPPOMM_VLLM_MEM
#                    HIPPOMM_VLM_URL
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${HIPPOMM_IMAGE:-ghcr.io/morph-graphs/hippomm:$(bash "$REPO/baselines/hippomm/docker/image_tag.sh" "$ARM")}
RUNTIME=${HIPPOMM_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${HIPPOMM_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${HIPPOMM_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/hippomm_$ARM.sif" ]; then
    IMAGE=$SIFS/hippomm_$ARM.sif
fi
HF=${HIPPOMM_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/hippomm/HippoMM/hippomm/core/hippocampal_memory.py" ] || { echo "run: git submodule update --init baselines/hippomm/HippoMM"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
# a folder that is not a git checkout (a synced copy) records an empty commit instead of stopping the run
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      HIPPOMM_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
      HIPPOMM_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)"
      HIPPOMM_UPSTREAM_COMMIT="$(git -C "$REPO/baselines/hippomm/HippoMM" rev-parse HEAD 2>/dev/null || true)")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: HippoMM's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export HIPPOMM_WORK=${HIPPOMM_WORK:-/work/hippomm}
fi
# every one of these, the key included, goes in by name: the value stays out of `ps`, files and logs
PASS=(OPENAI_API_KEY HF_TOKEN MERIT_HF_REPO HIPPOMM_TAG HIPPOMM_FETCH HIPPOMM_LIMIT HIPPOMM_VLLM_MEM HIPPOMM_VLM_URL
      HIPPOMM_OUT HIPPOMM_WORK)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, key included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/hippomm/run_hippomm.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "HIPPOMM_RUNTIME must be docker or apptainer"; exit 1; }
STAGE=$1
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
case $STAGE in
    prep|count|blind|upload) ;;   # no GPU
    *) GPU=${HIPPOMM_GPU:-${MERIT_GPU:?set HIPPOMM_GPU to the GPU index for this job}}
       # the container sees only the chosen GPU(s), renumbered from 0
       N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
       ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES="$SEEN") ;;
esac
exec docker run --rm --shm-size 32g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" \
    bash baselines/hippomm/run_hippomm.sh "$@"
