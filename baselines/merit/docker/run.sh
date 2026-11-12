#!/usr/bin/env bash
# Run run_merit.sh inside the arm's image: the morph_graph checkout is mounted at /repo (the code), the
# epic-morph checkout at /epic-morph (banks in, results out: outputs/v3/merit/ there).
#   MERIT_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/merit/docker/run.sh <stage> <video> <arm> [bank]
#   MERIT_GPU       GPU index(es) for this job, e.g. 0 or 2,3 (required: one H100 per m27 job)
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   MERIT_IMAGE     image (default ghcr.io/morph-graphs/merit:<arm>-<env hash>, from image_tag.sh)
#   MERIT_HF_CACHE  Hugging Face cache on the host (default ~/.cache/huggingface)
#   HDEPIC_VIDEOS   videos on the host, if not in $EPIC_MORPH_DIR/data/HD-EPIC/videos; mounted read-only
#   MERIT_OUT       optional, a path inside the container (e.g. /repo/baselines/merit/work/check)
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${MERIT_IMAGE:-ghcr.io/morph-graphs/merit:$(bash "$REPO/baselines/merit/docker/image_tag.sh" "$ARM")}
GPU=${MERIT_GPU:?set MERIT_GPU to the GPU index or indices for this job}
HF=${MERIT_HF_CACHE:-$HOME/.cache/huggingface}
[ -f "$REPO/baselines/merit/MERIT/eval/evaluate_egolife.py" ] || { echo "run: git submodule update --init baselines/merit/MERIT"; exit 1; }
mkdir -p "$HF"
VID=(); [ -n "${HDEPIC_VIDEOS:-}" ] && VID=(-v "$(cd "$HDEPIC_VIDEOS" && pwd):/videos:ro" -e HDEPIC_VIDEOS=/videos)
# the container sees only the chosen GPUs, renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" \
    -v "$REPO:/repo" -v "$EPIC:/epic-morph" -v "$HF:/hf" ${VID[@]+"${VID[@]}"} -w /repo \
    -e CUDA_VISIBLE_DEVICES="$SEEN" -e EPIC_MORPH_DIR=/epic-morph -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    ${MERIT_OUT:+-e MERIT_OUT="$MERIT_OUT"} \
    "$IMAGE" bash baselines/merit/run_merit.sh "$@"
