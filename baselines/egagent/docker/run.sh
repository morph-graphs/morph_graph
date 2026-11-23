#!/usr/bin/env bash
# Run run_egagent.sh inside the image (as MERIT's and WorldMM's docker/run.sh): the morph_graph checkout is
# mounted at /repo (the code; results go to baselines/egagent/results/ in it, then `upload` puts them in the
# Hugging Face dataset), the epic-morph checkout at /epic-morph (the banks), the videos read-only at /videos.
#   EPIC_MORPH_DIR=<epic-morph checkout> OPENAI_API_KEY=... bash baselines/egagent/docker/run.sh <stage> <video> <arm> [bank]
#   EPIC_MORPH_DIR   the epic-morph checkout (required)
#   OPENAI_API_KEY   for build (the memory, both arms) and for qa / blind of arm gpt41; or OPENROUTER_API_KEY
#                    without it: the same GPT-4.1 calls through OpenRouter (README "Through OpenRouter")
#   GEMINI_API_KEY   for qa / blind of arms gemini25pro and gemini36flash
#                    (GOOGLE_GEMINI_BASE_URL, passed on when set: another endpoint that speaks Google's API)
#                    Export a key in the shell: it is handed to the container by name, so it never appears on a
#                    command line, in a file or in the image. A paid stage without its key stops, nothing spent
#   HF_TOKEN         for upload and for taking a memory from the dataset; or `hf auth login` on the host (the
#                    cache, with its token, is mounted)
#   EGAGENT_RUNTIME  docker (default) or apptainer (clusters without Docker: the same image with `apptainer exec`)
#   EGAGENT_GPU      docker: GPU index for this job, e.g. 0 (the SigLIP 2 embedder; about 8 GB). Unset: CPU,
#                    fine for qa and blind, slow for build
#   EGAGENT_SERVER_IMAGE  arm q27 only: the image its Qwen3.8-27B server runs in (default MAGIC-Video's m27 image:
#                    ghcr.io/morph-graphs/magic:m27-<env hash>, or <EGAGENT_SIF_DIR>/magic_m27.sif with apptainer); needs
#                    one 141 GB GPU (H200) at the video's own resolution: 74 GB at a 30-frame caption call
#   EGAGENT_IMAGE    image (default ghcr.io/morph-graphs/egagent:gpu-<env hash>, from image_tag.sh); for
#                    apptainer, a .sif built once with `apptainer pull egagent_gpu.sif docker://<that image>`
#   EGAGENT_SIF_DIR  apptainer: a folder of such files (default MERIT_SIF_DIR); uses <dir>/egagent_gpu.sif when
#                    it exists and EGAGENT_IMAGE is unset
#   EGAGENT_HF_CACHE Hugging Face cache on the host (default MERIT_HF_CACHE, else ~/.cache/huggingface), mounted at /hf
#   MERIT_WORK       optional, the cluster's shared host work folder (MERIT's), mounted at /work; EGAgent's work
#                    folder is then /work/egagent (unless EGAGENT_WORK is set), group-writable
#   The job board (coord/worker.py) sets MERIT_GPU / MERIT_RUNTIME, read here when the EGAGENT_ ones are unset.
#   HDEPIC_VIDEOS    videos on the host (default $EPIC_MORPH_DIR/data/HD-EPIC/videos), mounted read-only at /videos
#   EGAGENT_OUT, EGAGENT_WORK  optional, paths inside the container (e.g. /repo/baselines/egagent/work/check)
#   passed through when set: MERIT_HF_REPO EGAGENT_TAG EGAGENT_FETCH EGAGENT_FRAME_PX EGAGENT_CHUNK_S THREADS LIMIT
#                    EGAGENT_QA_WORKERS
set -euo pipefail
: "${3:?usage: run.sh <stage> <video> <arm> [bank]}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
IMAGE=${EGAGENT_IMAGE:-ghcr.io/morph-graphs/egagent:$(bash "$REPO/baselines/egagent/docker/image_tag.sh" gpu)}
RUNTIME=${EGAGENT_RUNTIME:-${MERIT_RUNTIME:-docker}}
SIFS=${EGAGENT_SIF_DIR:-${MERIT_SIF_DIR:-}}
if [ "$RUNTIME" = apptainer ] && [ -z "${EGAGENT_IMAGE:-}" ] && [ -n "$SIFS" ] && [ -s "$SIFS/egagent_gpu.sif" ]; then
    IMAGE=$SIFS/egagent_gpu.sif
fi
HF=${EGAGENT_HF_CACHE:-${MERIT_HF_CACHE:-$HOME/.cache/huggingface}}
[ -f "$REPO/baselines/egagent/egagent/egagent/langgraph_agent.py" ] || { echo "run: git submodule update --init baselines/egagent/egagent"; exit 1; }
mkdir -p "$HF"
# the videos by their real host path: data/HD-EPIC is often a symlink, which would dangle inside the container
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
[ -d "$VIDEOS" ] || { echo "no videos at $VIDEOS (set HDEPIC_VIDEOS; epic-morph v3/DATA.md)"; exit 1; }
BINDS=("$REPO:/repo" "$EPIC:/epic-morph" "$HF:/hf" "$(cd -P "$VIDEOS" && pwd):/videos:ro")
# a folder that is not a git checkout (a synced copy) records the commit named in the environment
# (EGAGENT_REPO_COMMIT, EGAGENT_EPIC_COMMIT, EGAGENT_UPSTREAM_COMMIT), or an empty one, instead of stopping the run
ENVS=(EPIC_MORPH_DIR=/epic-morph HDEPIC_VIDEOS=/videos HF_HOME=/hf HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
      EGAGENT_REPO_COMMIT="${EGAGENT_REPO_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)}"
      EGAGENT_EPIC_COMMIT="${EGAGENT_EPIC_COMMIT:-$(git -C "$EPIC" rev-parse HEAD 2>/dev/null || true)}"
      EGAGENT_UPSTREAM_COMMIT="${EGAGENT_UPSTREAM_COMMIT:-$(git -C "$REPO/baselines/egagent/egagent" rev-parse HEAD 2>/dev/null || true)}")
if [ -n "${MERIT_WORK:-}" ]; then   # the shared work folder: EGAgent's own subfolder in it
    umask 002; mkdir -p "$MERIT_WORK"; BINDS+=("$(cd -P "$MERIT_WORK" && pwd):/work"); export EGAGENT_WORK=${EGAGENT_WORK:-/work/egagent}
fi
PASS=(OPENAI_API_KEY OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENAI_BASE_URL OPENAI_PROVIDER_NOTE OPENAI_MODEL_ALIASES EGAGENT_GPT_PREFIX EGAGENT_LOCAL_URL GEMINI_API_KEY GOOGLE_GEMINI_BASE_URL HF_TOKEN MERIT_HF_REPO EGAGENT_TAG EGAGENT_FETCH EGAGENT_FRAME_PX EGAGENT_CHUNK_S EGAGENT_MEM_WORKERS
      THREADS LIMIT EGAGENT_OUT EGAGENT_WORK EGAGENT_QA_WORKERS)
if [ "$RUNTIME" = apptainer ]; then
    case $IMAGE in *.sif) ;; *) IMAGE=docker://$IMAGE ;; esac
    # through APPTAINERENV_* (the environment), not --env: a command line, keys included, is visible in `ps`
    export "${ENVS[@]/#/APPTAINERENV_}"
    for v in "${PASS[@]}" CUDA_VISIBLE_DEVICES; do [ -n "${!v:-}" ] && export "APPTAINERENV_$v=${!v}"; done
    case "$3 $1" in "q27 build"|"q27 qa"|"q27 blind"|"q27 all") SERVE=1 ;; *) SERVE= ;; esac
    if [ -n "$SERVE" ] && [ -z "${EGAGENT_LOCAL_URL:-}" ]; then
        # arm q27: Qwen3.8-27B behind MAGIC-Video's server (../magic/epic/hf_server.py), from MAGIC's m27 image (its
        # transformers 5.8 environment), on this job's GPUs and the host's network; EGAgent's container calls it
        SIMG=${EGAGENT_SERVER_IMAGE:-$SIFS/magic_m27.sif}; PORT=${EGAGENT_PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 20000))}
        [ -s "$SIMG" ] || { echo "arm q27 needs MAGIC-Video's m27 image for its server: $SIMG (or EGAGENT_SERVER_IMAGE)"; exit 1; }
        SLOG=$(mktemp "${TMPDIR:-/tmp}/egagent_server_$2_XXXX.log")
        DMAP=cuda:0; [ "$(nvidia-smi -L | wc -l)" -le 1 ] || DMAP=auto   # several GPUs: the model spread over all of them
        APPTAINERENV_PORT=$PORT APPTAINERENV_DMAP=$DMAP apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" "$SIMG" bash -c \
            'exec "$MAGIC_SERVER_PY" baselines/magic/epic/hf_server.py --model Qwen/Qwen3.8-27B --name Qwen/Qwen3.8-27B --port "$PORT" --max_tokens 8192 --token_budget 96000 --solo_images 50 --device_map "$DMAP"' > "$SLOG" 2>&1 &
        SPID=$!; trap 'kill $SPID 2>/dev/null || true' EXIT
        for i in $(seq 1 240); do
            curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
            kill -0 $SPID 2>/dev/null || { echo "Qwen server died:"; tail -30 "$SLOG"; exit 1; }
            sleep 5
        done
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "Qwen server not up after 20 min"; tail -30 "$SLOG"; exit 1; }
        echo "Qwen server up after $((i * 5)) s on port $PORT (log $SLOG): $(grep -m1 loaded "$SLOG")"
        export APPTAINERENV_EGAGENT_LOCAL_URL=http://127.0.0.1:$PORT/v1
        rc=0; apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
            "$IMAGE" bash baselines/egagent/run_egagent.sh "$@" || rc=$?
        echo "Qwen server: $(curl -sf "http://127.0.0.1:$PORT/v1/models" | head -c 400)"
        exit $rc
    fi
    exec apptainer exec --nv --cleanenv --pwd /repo --bind "$(IFS=,; echo "${BINDS[*]}")" \
        "$IMAGE" bash baselines/egagent/run_egagent.sh "$@"
fi
[ "$RUNTIME" = docker ] || { echo "EGAGENT_RUNTIME must be docker or apptainer"; exit 1; }

ARGS=(); for b in "${BINDS[@]}"; do ARGS+=(-v "$b"); done
for e in "${ENVS[@]}"; do ARGS+=(-e "$e"); done
for v in "${PASS[@]}"; do [ -n "${!v:-}" ] && ARGS+=(-e "$v"); done   # by name: the value stays out of `ps`
GPU=${EGAGENT_GPU:-${MERIT_GPU:-}}
if [ -n "$GPU" ]; then   # the container sees only the chosen GPU(s), renumbered from 0
    N=$(echo "$GPU" | tr ',' '\n' | grep -c .); ARGS+=(--gpus "\"device=$GPU\"" -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N - 1)))")
fi
case "$3 $1" in "q27 build"|"q27 qa"|"q27 blind"|"q27 all") SERVE=1 ;; *) SERVE= ;; esac
if [ -n "$SERVE" ] && [ -z "${EGAGENT_LOCAL_URL:-}" ]; then
    # arm q27, as on the apptainer route above: Qwen3.8-27B behind MAGIC-Video's server in MAGIC's m27 image, on this
    # job's GPUs; both containers on the host's network, so EGAgent's reaches it at 127.0.0.1
    [ -n "$GPU" ] || { echo "arm q27 needs a GPU: set EGAGENT_GPU"; exit 1; }
    SIMG=${EGAGENT_SERVER_IMAGE:-ghcr.io/morph-graphs/magic:$(bash "$REPO/baselines/magic/docker/image_tag.sh" m27)}
    PORT=${EGAGENT_PORT:-$((20000 + $$ % 20000))}; SNAME=egagent-qwen-$$
    docker run -d --rm --name "$SNAME" --network host --gpus "\"device=$GPU\"" --shm-size 32g --user "$(id -u):$(id -g)" -w /repo \
        -v "$REPO:/repo" -v "$HF:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" ${HF_TOKEN:+-e HF_TOKEN} "$SIMG" bash -c \
        'exec "$MAGIC_SERVER_PY" baselines/magic/epic/hf_server.py --model Qwen/Qwen3.8-27B --name Qwen/Qwen3.8-27B --port '"$PORT"' --max_tokens 8192 --token_budget 96000 --solo_images 50 --device_map '"$( [ "${GPU/,/}" = "$GPU" ] && echo cuda:0 || echo auto)" >/dev/null
    trap 'docker stop "$SNAME" >/dev/null 2>&1 || true' EXIT
    for i in $(seq 1 360); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
        docker inspect "$SNAME" >/dev/null 2>&1 || { echo "Qwen server died (docker logs are gone with it: rerun its command by hand to see why)"; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "Qwen server not up after 30 min:"; docker logs --tail 30 "$SNAME"; exit 1; }
    echo "Qwen server up after $((i * 5)) s on port $PORT (container $SNAME)"
    rc=0; docker run --rm --network host --shm-size 8g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" \
        -e EGAGENT_LOCAL_URL="http://127.0.0.1:$PORT/v1" "$IMAGE" bash baselines/egagent/run_egagent.sh "$@" || rc=$?
    exit $rc
fi
exec docker run --rm --shm-size 8g --user "$(id -u):$(id -g)" -w /repo "${ARGS[@]}" "$IMAGE" bash baselines/egagent/run_egagent.sh "$@"
