#!/usr/bin/env bash
# Run run_direct.sh inside the image of the arm's model (as MERIT's docker/run.sh). Direct has no image of its own:
# a model runs in MERIT's image of its environment (image_tag.sh here: qwen3vl-8b in merit:m8-..., qwen38-27b in
# merit:m27-...), so a machine that runs MERIT has everything. The morph_graph checkout is mounted at /repo (the
# code; results go to baselines/direct/results/ in it, then `upload` puts them in the Hugging Face dataset), the
# epic-morph checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or `hf auth login` on the host
# (the cache, with its token, is mounted).
#   DIRECT_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/direct/docker/run.sh <stage> <video> <arm> [bank]
#   DIRECT_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image, run with
#                   `apptainer exec`; GPUs are the ones the scheduler gave the job, so DIRECT_GPU is not needed)
#   DIRECT_GPU      docker: GPU index(es) for this job, e.g. 0, or 0,1 (the 27B spreads over the GPUs it is given)
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   DIRECT_IMAGE    image (default ghcr.io/morph-graphs/merit:<m8|m27>-<env hash>); for apptainer, a .sif
#   DIRECT_SIF_DIR  apptainer: a folder of .sif files (default MERIT_SIF_DIR); uses MERIT's <dir>/merit_<m8|m27>.sif
#                   when it exists and DIRECT_IMAGE is unset
#   DIRECT_HF_CACHE Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   MERIT_WORK      optional, the cluster's shared host work folder (MERIT's), mounted at /work: the 1 fps frames are
#                   MERIT's prep, cut once for both (default: baselines/merit/work in the checkout)
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the DIRECT_ ones are unset.
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   DIRECT_OUT      optional, a path inside the container (e.g. /repo/baselines/direct/work/check)
#   DIRECT_REPO_COMMIT, DIRECT_MERIT_COMMIT, DIRECT_EPIC_COMMIT  for runs.jsonl when a checkout has no .git (a synced
#                   copy); read with git otherwise
#   passed through when set: HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX DIRECT_LIMIT THREADS
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
TAG=$(bash "$REPO/baselines/direct/docker/image_tag.sh" "$ARM"); ENV=${TAG%%-*}   # MERIT's environment of the model
IMAGE=${DIRECT_IMAGE:-ghcr.io/morph-graphs/merit:$TAG}
RUNTIME=${DIRECT_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${DIRECT_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${DIRECT_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/merit_$ENV.sif" ]; then
    IMAGE=$SIFS/merit_$ENV.sif
fi
HF=${DIRECT_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/merit/MERIT/eval/evaluate_egolife.py" ] || { echo "run: git submodule update --init baselines/merit/MERIT"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
REPO_COMMIT=${DIRECT_REPO_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}
MERIT_COMMIT=${DIRECT_MERIT_COMMIT:-$(git -C "$REPO/baselines/merit/MERIT" rev-parse HEAD)}
EPIC_COMMIT=${DIRECT_EPIC_COMMIT:-$(git -C "$EPIC" rev-parse HEAD)}
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      DIRECT_REPO_COMMIT="$REPO_COMMIT" DIRECT_MERIT_COMMIT="$MERIT_COMMIT" DIRECT_EPIC_COMMIT="$EPIC_COMMIT")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder, group-writable as MERIT's run.sh makes it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); ENVS+=(MERIT_WORK=/work)
fi
PASS=(HF_TOKEN MERIT_HF_REPO MERIT_OUTBOX DIRECT_LIMIT DIRECT_OUT THREADS)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/direct/run_direct.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "DIRECT_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${DIRECT_GPU:-${MERIT_GPU:?set DIRECT_GPU to the GPU index or indices for this job}}
# the container sees only the chosen GPUs, renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" -w /repo \
    -e CUDA_VISIBLE_DEVICES="$SEEN" "${ARGS[@]}" "$IMAGE" bash baselines/direct/run_direct.sh "$@"
