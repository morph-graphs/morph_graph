#!/usr/bin/env bash
# Run run_egor1.sh inside the image (as the other baselines' docker/run.sh): the morph_graph checkout is mounted
# at /repo (the code; results go to baselines/egor1/results/ in it, then `upload` puts them in the Hugging Face
# dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos.
#   EGOR1_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/egor1/docker/run.sh <stage> <video> <arm> [bank]
#
# Keys: export them in the shell that starts this script. They are handed to the container by name (docker
# `-e NAME`, apptainer APPTAINERENV_NAME), so a key never appears on a command line, in a file, in the image or
# in a log. A paid stage without its keys exits with code 5 before calling anything.
#   GEMINI_API_KEY   Gemini 2.5 Pro: the captions (build) and video_llm (qa)
#   OPENAI_API_KEY   GPT-4.1 and GPT-4o (build, qa); or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
#   HF_TOKEN         upload, and build's look in the results dataset; or `hf auth login` on the host (the cache,
#                    with its token, is mounted)
#
#   EGOR1_RUNTIME    docker (default) or apptainer (clusters without Docker: the same image with `apptainer exec`;
#                    the GPU is the one the scheduler gave the job, so EGOR1_GPU is not needed)
#   EGOR1_GPU        docker: the GPU index for qa (one GPU, ~8 GB). prep, build and upload need none
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   EGOR1_IMAGE      image (default ghcr.io/morph-graphs/egor1:<arm>-<env hash>, from image_tag.sh); for apptainer
#                    a .sif built once with `apptainer pull egor1_<arm>.sif docker://<that image>`
#   EGOR1_SIF_DIR    apptainer: a folder of such files (default MERIT_SIF_DIR); <dir>/egor1_<arm>.sif is used when
#                    it exists and EGOR1_IMAGE is unset
#   EGOR1_HF_CACHE   Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted
#                    at /hf: the agent's weights (Ego-R1/Ego-R1-Agent-3B, 13 GB) are downloaded there on first use
#   MERIT_WORK       optional, the cluster's shared host work folder, mounted at /work; Ego-R1's work folder is
#                    then /work/egor1 (unless EGOR1_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the EGOR1_ ones are unset.
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   EGOR1_OUT, EGOR1_WORK  optional, paths inside the container (e.g. /repo/baselines/egor1/work/check)
#   passed through when set: MERIT_HF_REPO EGOR1_TAG EGOR1_FETCH EGOR1_LIMIT EGOR1_AGENT THREADS HF_HUB_OFFLINE
set -euo pipefail
STAGE=${1:?usage: run.sh <stage> <video> <arm> [bank]}; ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${EGOR1_IMAGE:-ghcr.io/morph-graphs/egor1:$(bash "$REPO/baselines/egor1/docker/image_tag.sh" "$ARM")}
RUNTIME=${EGOR1_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${EGOR1_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${EGOR1_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/egor1_$ARM.sif" ]; then
    IMAGE=$SIFS/egor1_$ARM.sif
fi
HF=${EGOR1_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/egor1/Ego-R1/api/rag/r1rag/ragagent.py" ] || { echo "run: git submodule update --init baselines/egor1/Ego-R1"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
# a folder that is not a git checkout (a synced copy) records an empty commit instead of stopping the run
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf
      EGOR1_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
      EGOR1_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)"
      EGOR1_UPSTREAM_COMMIT="$(git -C "$REPO/baselines/egor1/Ego-R1" rev-parse HEAD 2>/dev/null || true)")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: Ego-R1's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export EGOR1_WORK=${EGOR1_WORK:-/work/egor1}
fi
KEYS=(GEMINI_API_KEY OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY HF_TOKEN)
PASS=(MERIT_HF_REPO EGOR1_TAG EGOR1_FETCH EGOR1_LIMIT EGOR1_AGENT EGOR1_OUT EGOR1_WORK THREADS HF_HUB_OFFLINE)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${KEYS[@]}" "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/egor1/run_egor1.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "EGOR1_RUNTIME must be docker or apptainer"; exit 1; }
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${KEYS[@]}" "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
GPU=${EGOR1_GPU:-${MERIT_GPU:-}}
case $STAGE in
    prep|build|upload) ;;                         # no GPU: ffmpeg and the APIs
    *) [ -n "$GPU" ] || { echo "set EGOR1_GPU to the GPU index for this job"; exit 1; }
       ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES=0) ;;
esac
exec docker run --rm --shm-size 8g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" bash baselines/egor1/run_egor1.sh "$@"
