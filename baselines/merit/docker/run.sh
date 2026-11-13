#!/usr/bin/env bash
# Run run_merit.sh inside the arm's image: the morph_graph checkout is mounted at /repo (the code; results go
# to baselines/merit/results/ in it, then `upload` puts them in the Hugging Face dataset), the epic-morph
# checkout at /epic-morph (the banks). The Hugging Face token: HF_TOKEN, or `hf auth login` on the host
# (the cache, with its token, is mounted).
#   MERIT_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/merit/docker/run.sh <stage> <video> <arm> [bank]
#   MERIT_RUNTIME   docker (default) or apptainer (clusters without Docker: the same image, run with
#                   `apptainer exec`; GPUs are the ones the scheduler gave the job, so MERIT_GPU is not needed)
#   MERIT_GPU       docker: GPU index(es) for this job, e.g. 0 or 2,3 (required: one H100 per m27 job)
#   EPIC_MORPH_DIR  the epic-morph checkout (required)
#   MERIT_IMAGE     image (default ghcr.io/morph-graphs/merit:<arm>-<env hash>, from image_tag.sh); for
#                   apptainer, a .sif built once with `apptainer pull merit_<arm>.sif docker://<that image>`
#   MERIT_SIF_DIR   apptainer: a folder of such files; uses <dir>/merit_<arm>.sif when it exists and MERIT_IMAGE
#                   is unset, so one worker can run tasks of both arms
#   MERIT_HF_CACHE  Hugging Face cache on the host (default ~/.cache/huggingface)
#   HDEPIC_VIDEOS   videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   MERIT_WORK      optional, a host folder for clips, frames, run folders and caches (default: baselines/merit/work
#                   in the checkout), mounted at /work. One folder per cluster, shared by every worker there, so a
#                   video is cut once (prep holds a per-video lock) and no checkout fills its owner's quota; files
#                   are made group-writable (umask 002) so the other workers' users can reuse and reset them
#   MERIT_OUT       optional, a path inside the container (e.g. /repo/baselines/merit/work/check)
#   MERIT_CONTEXT   optional, blind only: a context file, as a path inside the container (under /repo or /epic-morph); needs MERIT_TAG
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${MERIT_IMAGE:-ghcr.io/morph-graphs/merit:$(bash "$REPO/baselines/merit/docker/image_tag.sh" "$ARM")}
if [ -z "${MERIT_IMAGE:-}" ] && [ -n "${MERIT_SIF_DIR:-}" ] && [ -s "$MERIT_SIF_DIR/merit_$ARM.sif" ]; then
    IMAGE=$MERIT_SIF_DIR/merit_$ARM.sif
fi
RUNTIME=${MERIT_RUNTIME:-docker}
HF=${MERIT_HF_CACHE:-$HOME/.cache/huggingface}
[ -f "$REPO/baselines/merit/MERIT/eval/evaluate_egolife.py" ] || { echo "run: git submodule update --init baselines/merit/MERIT"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
if [ -n "${MERIT_WORK:-}" ]; then
    umask 002; mkdir -p "$MERIT_WORK"; WORK=$(cd -P "$MERIT_WORK" && pwd)
fi
COMMITS=(MERIT_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)" MERIT_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD)"
         MERIT_SUBMODULE_COMMIT="$(git -C "$REPO/baselines/merit/MERIT" rev-parse HEAD)")
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, token included, is visible in `ps`
    export APPTAINERENV_EPIC_MORPH_DIR=/epic-morph APPTAINERENV_HDEPIC_VIDEOS=/videos APPTAINERENV_HF_HOME=/hf
    export APPTAINERENV_HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} "${COMMITS[@]/#/APPTAINERENV_}"
    [ -n "${WORK:-}" ] && export APPTAINERENV_MERIT_WORK=/work
    for v in HF_TOKEN MERIT_HF_REPO MERIT_TAG MERIT_NO_FRAMES MERIT_FETCH MERIT_OUT MERIT_CONTEXT CUDA_VISIBLE_DEVICES; do
        [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"
    done
    exec apptainer exec --nv --cleanenv --pwd /repo \
        --bind "$REPO:/repo,$EPIC:/epic-morph,$HF:/hf,$(cd -P "$VIDEOS" && pwd):/videos:ro${WORK:+,$WORK:/work}" \
        "$IMAGE" bash baselines/merit/run_merit.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "MERIT_RUNTIME must be docker or apptainer"; exit 1; }
GPU=${MERIT_GPU:?set MERIT_GPU to the GPU index or indices for this job}
# the container sees only the chosen GPUs, renumbered from 0
N=$(echo "$GPU" | tr ',' '\n' | grep -c .); SEEN=$(seq -s, 0 $((N - 1)))
exec docker run --rm --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" \
    -v "$REPO:/repo" -v "$EPIC:/epic-morph" -v "$HF:/hf" -v "$(cd -P "$VIDEOS" && pwd):/videos:ro" -w /repo \
    ${WORK:+-v "$WORK:/work" -e MERIT_WORK=/work} \
    -e CUDA_VISIBLE_DEVICES="$SEEN" -e EPIC_MORPH_DIR=/epic-morph -e HDEPIC_VIDEOS=/videos -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    ${HF_TOKEN:+-e HF_TOKEN} ${MERIT_HF_REPO:+-e MERIT_HF_REPO} \
    ${MERIT_TAG:+-e MERIT_TAG} ${MERIT_NO_FRAMES:+-e MERIT_NO_FRAMES} ${MERIT_FETCH:+-e MERIT_FETCH} \
    -e "${COMMITS[0]}" -e "${COMMITS[1]}" -e "${COMMITS[2]}" \
    ${MERIT_OUT:+-e MERIT_OUT="$MERIT_OUT"} ${MERIT_CONTEXT:+-e MERIT_CONTEXT="$MERIT_CONTEXT"} \
    "$IMAGE" bash baselines/merit/run_merit.sh "$@"
