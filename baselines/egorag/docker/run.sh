#!/usr/bin/env bash
# Run run_egorag.sh inside the image (as MERIT's and WorldMM's docker/run.sh): the morph_graph checkout is
# mounted at /repo (the code; results go to baselines/egorag/results/ in it, then `upload` puts them in the
# Hugging Face dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos.
#   EGORAG_GPU=0 EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/egorag/docker/run.sh <stage> <video> <arm> [bank]
#   OPENAI_API_KEY, DEEPSEEK_API_KEY   the paid stages' keys (build's summaries: both; qa: OPENAI_API_KEY).
#                    Export them in the shell: they are handed to the container by name, so they never appear
#                    on a command line, in a file or in the image. Upstream's Azure variables pass the same way.
#   EGORAG_RUNTIME   docker (default) or apptainer (clusters without Docker: the same image, run with
#                    `apptainer exec`; GPUs are the ones the scheduler gave the job, so EGORAG_GPU is not needed)
#   EGORAG_GPU       docker: the GPU index for this job (one 48 GB GPU for build's captions; qa needs none
#                    and runs without it)
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   EGORAG_IMAGE     image (default ghcr.io/morph-graphs/egorag:<arm>-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull egorag_<arm>.sif docker://<that image>`
#   EGORAG_SIF_DIR   apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/egorag_<arm>.sif when
#                    it exists and EGORAG_IMAGE is unset, so one board worker runs every baseline and arm
#   EGORAG_HF_CACHE  Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted
#                    at /hf: EgoGPT-7b, SigLIP so400m and the OpenCLIP ViT-B-32 text encoder are downloaded into it
#   MERIT_WORK       optional, the cluster's shared host work folder (MERIT's), mounted at /work; EgoRAG's work
#                    folder is then /work/egorag (unless EGORAG_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the EGORAG_ ones are unset.
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   EGORAG_OUT, EGORAG_WORK  optional, paths inside the container (e.g. /repo/baselines/egorag/work/check)
#   passed through when set: HF_TOKEN MERIT_HF_REPO EGORAG_TAG EGORAG_LIMIT EGORAG_FETCH EGORAG_GPT_MODEL
#                    EGORAG_DEEPSEEK_MODEL EGORAG_EGOGPT OPENAI_BASE_URL DEEPSEEK_BASE_URL DEPLOYMENT_NAME THREADS
set -euo pipefail
ARM=${3:?usage: run.sh <stage> <video> <arm> [bank]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${EGORAG_IMAGE:-ghcr.io/morph-graphs/egorag:$(bash "$REPO/baselines/egorag/docker/image_tag.sh" "$ARM")}
RUNTIME=${EGORAG_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${EGORAG_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${EGORAG_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/egorag_$ARM.sif" ]; then
    IMAGE=$SIFS/egorag_$ARM.sif
fi
HF=${EGORAG_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/egorag/EgoLife/EgoRAG/egorag/agents/RagAgent.py" ] || { echo "run: git submodule update --init baselines/egorag/EgoLife"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
# a folder that is not a git checkout (a synced copy) records an empty commit instead of stopping the run
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      EGORAG_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
      EGORAG_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)"
      EGORAG_UPSTREAM_COMMIT="$(git -C "$REPO/baselines/egorag/EgoLife" rev-parse HEAD 2>/dev/null || true)")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: EgoRAG's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export EGORAG_WORK=${EGORAG_WORK:-/work/egorag}
fi
# keys and settings, by name only: a value never appears on a command line
PASS=(OPENAI_API_KEY DEEPSEEK_API_KEY AZURE_API_KEY AZURE_API_ENDPOINT AZURE_DEEPSEEK_KEY AZURE_DEEPSEEK_ENDPOINT
      DEPLOYMENT_NAME OPENAI_BASE_URL DEEPSEEK_BASE_URL HF_TOKEN MERIT_HF_REPO EGORAG_TAG EGORAG_LIMIT EGORAG_FETCH
      EGORAG_GPT_MODEL EGORAG_DEEPSEEK_MODEL EGORAG_EGOGPT EGORAG_OUT EGORAG_WORK THREADS)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    # apptainer shares the host's /tmp: a per-user numba cache, not the image's /tmp/numba
    export APPTAINERENV_NUMBA_CACHE_DIR=/tmp/numba_$(id -u)
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/egorag/run_egorag.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "EGORAG_RUNTIME must be docker or apptainer"; exit 1; }
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
GPU=${EGORAG_GPU:-${MERIT_GPU:-}}
if [ -n "$GPU" ]; then   # the container sees only the chosen GPU(s), renumbered from 0
    N=$(echo "$GPU" | tr ',' '\n' | grep -c .); ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N - 1)))")
else
    case $1 in qa|upload) ;; *) echo "set EGORAG_GPU to the GPU index for this job (build's captions need one 48 GB GPU)"; exit 1 ;; esac
fi
exec docker run --rm --shm-size 16g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" \
    bash baselines/egorag/run_egorag.sh "$@"
