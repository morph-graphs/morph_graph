#!/usr/bin/env bash
# Run run_m3agent.sh inside the image (as MERIT's docker/run.sh): the morph_graph checkout is mounted at /repo
# (the code; results go to baselines/m3agent/results/ in it, then `upload` puts them in the Hugging Face
# dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos, the models at
# /models. The stage `models` downloads the models into M3AGENT_MODELS first (fetch_models.sh; no GPU).
#   M3AGENT_MODELS=<dir> bash baselines/m3agent/docker/run.sh models
#   M3AGENT_GPU=0 M3AGENT_MODELS=<dir> EPIC_MORPH_DIR=<epic-morph checkout> bash baselines/m3agent/docker/run.sh <stage> <video> m3 [bank]
# The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME / MERIT_SIF_DIR, read here as fallbacks.
#   M3AGENT_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image, run with
#                    `apptainer exec`; GPUs are the ones the scheduler gave the job, so M3AGENT_GPU is not needed)
#   M3AGENT_GPU      docker: GPU index(es) for this job: one >= 40 GB GPU for build, 2x 48 GB (or one 80 GB with
#                    M3AGENT_TP=1) for qa and blind
#   M3AGENT_MODELS   the models folder on the host (required), mounted at /models (writable: `models` fills it)
#   EPIC_MORPH_DIR   the epic-morph checkout (required, except for `models`)
#   M3AGENT_BACKEND  local (default: faster-whisper and Qwen3-Embedding stand in for the closed APIs) or api
#                    (upstream's Gemini speech segments and OpenAI text-embedding-3-large)
#   GEMINI_API_KEY, OPENAI_API_KEY  the API keys for M3AGENT_BACKEND=api. Export them in the shell: they are
#                    handed to the container by name, so they never appear on a command line, in a file or in the image
#   HF_TOKEN         for upload; or `hf auth login` on the host (the cache, with its token, is mounted)
#   M3AGENT_IMAGE    image (default ghcr.io/morph-graphs/m3agent:m3-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull m3agent_m3.sif docker://<that image>`
#   M3AGENT_SIF_DIR  apptainer: a folder of such files; uses <dir>/m3agent_m3.sif when it exists and M3AGENT_IMAGE is unset
#   M3AGENT_HF_CACHE Hugging Face cache on the host (default ~/.cache/huggingface), mounted at /hf
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos)
#   M3AGENT_WORK     optional, a host folder for clips, intermediate outputs and control runs (default
#                    baselines/m3agent/work in the checkout), mounted at /work; files made group-writable (umask 002)
#   passed through when set: M3AGENT_TAG M3AGENT_FETCH M3AGENT_TP M3AGENT_GPU_UTIL M3AGENT_MAX_LEN
#                    M3AGENT_EMBED_DEVICE M3AGENT_MEM_RETRIES THREADS MERIT_HF_REPO M3AGENT_OUT (a path inside the container)
set -euo pipefail
STAGE=${1:?usage: run.sh <models|prep|build|qa|blind|upload|all> [<video> m3 [bank]]}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMAGE=${M3AGENT_IMAGE:-ghcr.io/morph-graphs/m3agent:$(bash "$REPO/baselines/m3agent/docker/image_tag.sh" m3)}
SIF_DIR=${M3AGENT_SIF_DIR:-${MERIT_SIF_DIR:-}}
[ -z "${M3AGENT_IMAGE:-}" ] && [ -n "$SIF_DIR" ] && [ -s "$SIF_DIR/m3agent_m3.sif" ] && IMAGE=$SIF_DIR/m3agent_m3.sif
RUNTIME=${M3AGENT_RUNTIME:-${MERIT_RUNTIME:-docker}}
HF=${M3AGENT_HF_CACHE:-$HOME/.cache/huggingface}
umask 002; mkdir -p "$HF" "${M3AGENT_MODELS:?set M3AGENT_MODELS to the models folder on the host}"
BINDS=("$REPO:/repo" "$HF:/hf" "$(cd -P "$M3AGENT_MODELS" && pwd):/models")
ENVS=(HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" M3AGENT_MODELS=/models)
PASS=(HF_TOKEN MERIT_HF_REPO GEMINI_API_KEY OPENAI_API_KEY M3AGENT_BACKEND M3AGENT_TAG M3AGENT_FETCH M3AGENT_TP
      M3AGENT_GPU_UTIL M3AGENT_MAX_LEN M3AGENT_EMBED_DEVICE M3AGENT_MEM_RETRIES THREADS M3AGENT_OUT)
if [ "$STAGE" = models ]; then
    CMD=(bash baselines/m3agent/fetch_models.sh /models)
else
    : "${3:?usage: run.sh <stage> <video> m3 [bank]}"
    [ -f "$REPO/baselines/m3agent/m3-agent/m3_agent/control.py" ] || { echo "run: git submodule update --init baselines/m3agent/m3-agent"; exit 1; }
    EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
    # the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
    VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
    [ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
    BINDS+=("$EPIC:/epic-morph" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
    ENVS+=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos
           M3AGENT_REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)" M3AGENT_EPIC_COMMIT="$(git -C "$EPIC" rev-parse HEAD)"
           M3AGENT_SUBMODULE_COMMIT="$(git -C "$REPO/baselines/m3agent/m3-agent" rev-parse HEAD)")
    if [ -n "${M3AGENT_WORK:-}" ]; then
        mkdir -p "$M3AGENT_WORK"; BINDS+=("$(cd -P "$M3AGENT_WORK" && pwd):/work"); ENVS+=(M3AGENT_WORK=/work)
    fi
    CMD=(bash baselines/m3agent/run_m3agent.sh "$@")
fi
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    # apptainer shares the host's /tmp: per-user caches, not the image's /tmp/triton and /tmp/numba
    export APPTAINERENV_TRITON_CACHE_DIR=/tmp/triton_$(id -u) APPTAINERENV_NUMBA_CACHE_DIR=/tmp/numba_$(id -u)
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" "$IMAGE" "${CMD[@]}"
fi
[ "$RUNTIME" = docker ] || { echo "M3AGENT_RUNTIME must be docker or apptainer"; exit 1; }
ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
if [ "$STAGE" != models ]; then
    GPU=${M3AGENT_GPU:-${MERIT_GPU:?set M3AGENT_GPU to the GPU index or indices for this job}}
    # the container sees only the chosen GPUs, renumbered from 0
    N=$(echo "$GPU" | tr ',' '\n' | grep -c .)
    ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N - 1)))")
fi
exec docker run --rm --shm-size 32g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" "${CMD[@]}"
