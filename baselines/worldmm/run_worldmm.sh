#!/usr/bin/env bash
# WorldMM on one HD-EPIC video, without SLURM (the EPIC port of Vatsal's batch10 adapter). See README.md here.
#
#   bash baselines/worldmm/run_worldmm.sh <stage> <video> <arm> [bank]
#
#   stage  prep   512 px copy of the video + WorldMM's 10 s segment skeleton                CPU, ffmpeg
#          build  WorldMM's memory of the whole video: 10 s captions, multiscale captions,   GPU
#                 episodic / semantic / visual memory, HippoRAG OpenIE (no questions needed)
#          qa     WorldMM answers every question of <bank> for this video, each from the     GPU (needs build)
#                 memory up to its t_ask
#          blind  WorldMM's answer prompt with no context: the no-memory floor               GPU (no build needed)
#          upload this video and arm's results folder to the Hugging Face dataset            (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    m8   Qwen3-VL-8B-Instruct for every role (captions, memory, retrieval, answers)   one GPU with >= 40 GB
#          m27  Qwen3.8-27B (instruct mode) for every role                                  one 80 GB GPU, or 2x 48 GB
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing; byte-identical for anyone at that commit)
#
# Environment (see README.md): PY  python of the client environment (env/setup_env.sh client): WorldMM itself,
#                                  and the backbone server of m8
#   SERVER_PY        python of the backbone server; default PY (m8); m27 needs env/setup_env.sh server27
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   WORLDMM_MODEL_DIR  a folder of model folders (Qwen3-VL-8B-Instruct, Qwen3.8-27B, Qwen3-Embedding-4B,
#                    VLM2Vec-V2.0, ...); unset: Hugging Face ids (hub or cache)
#   WORLDMM_WORK     default baselines/worldmm/work: 512 px videos, skeletons, build intermediates, question
#                    files, per-question rows, HippoRAG caches, logs (not committed)
#   WORLDMM_OUT      default baselines/worldmm/results: per <arm>/<video> the memory files, answers_<bank>.json.gz,
#                    blind_<bank>.json and runs.jsonl (not committed; `upload` puts them in the dataset
#                    morph-graphs/epic-morph-results at worldmm/<arm>/<video>/)
#   WORLDMM_TAG      a variant name for the answers (answers_<bank>_<tag>.json.gz), so variants never overwrite
#   WORLDMM_FETCH    0 to never take the memory from the dataset (build normally reuses it when it is there)
#   WORLDMM_REPO_COMMIT, WORLDMM_SUBMODULE_COMMIT, WORLDMM_EPIC_COMMIT  commits for runs.jsonl when a checkout
#                    has no .git (a synced copy); read with git otherwise
# Every step skips finished work, so rerunning the same command resumes. The memory is built once per video
# and arm and shared by every bank and variant.
set -euo pipefail
STAGE=${1:?usage: run_worldmm.sh <prep|build|qa|blind|upload|all> <video> <m8|m27> [bank]}
V=${2:?video id}; ARM=${3:?arm m8 or m27}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/WorldMM   # R: the submodule
PY=${PY:?set PY to the python of the WorldMM client environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/src/worldmm/memory/memory.py" ] || { echo "WorldMM submodule missing: git submodule update --init baselines/worldmm/WorldMM"; exit 1; }
case $ARM in
    m8)  MODEL=qwen3vl-8b; HF=Qwen/Qwen3-VL-8B-Instruct; SERVER_PY=${SERVER_PY:-$PY} ;;
    m27) MODEL=qwen38-27b; HF=Qwen/Qwen3.8-27B; SERVER_PY=${SERVER_PY:?m27: set SERVER_PY to the python of env/setup_env.sh server27} ;;
    *) echo "arm must be m8 or m27"; exit 1 ;;
esac
MP=$HF; [ -n "${WORLDMM_MODEL_DIR:-}" ] && [ -d "$WORLDMM_MODEL_DIR/${HF#*/}" ] && MP=$WORLDMM_MODEL_DIR/${HF#*/}
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${WORLDMM_WORK:-$M/work}; OUTROOT=${WORLDMM_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; SKEL=$WORK/skeleton/$V/10sec_skeleton.json; SMALL=$WORK/videos_512/$V.mp4
BUILD=$WORK/build/$ARM; TAG=${WORLDMM_TAG:-}; SUF=${TAG:+_$TAG}
# the memory's files (the same list as epic/wmm_paths.memory_files)
MEMORY=("{video}_10sec.json" "{video}_30sec.json" "{video}_3min.json" "{video}_10min.json" "{video}_semantic.json"
        "{video}_visual.pkl" "{video}_openie_10sec.json" "{video}_openie_30sec.json" "{video}_openie_3min.json"
        "{video}_openie_10min.json")
mkdir -p "$OUT" "$WORK/logs"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODEL EPIC
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
have_memory() { local m; for m in "${MEMORY[@]}"; do [ -s "$OUT/${m//\{video\}/$V}" ] || return 1; done; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
commit = os.environ.get("WORLDMM_REPO_COMMIT") or git("rev-parse", "HEAD")
sub = os.environ.get("WORLDMM_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/worldmm/WorldMM")
bank = os.environ.get("WORLDMM_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/worldmm"))
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "worldmm_commit": sub, "epic_morph_commit": bank, "model": os.environ.get("MODEL"),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "server_transformers": os.environ.get("SERVER_TRANSFORMERS"), "gpus": gpus,
                  "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
cd "$REPO"

# GPUs: the backbone server on the first GPU (m27 on 2+ GPUs: spread over all of them, leaving room on the
# last one), WorldMM itself (Qwen3-Embedding-4B + VLM2Vec, ~13 GB) on the last; one GPU holds both for m8
GPUS=${CUDA_VISIBLE_DEVICES:-$( (nvidia-smi -L 2>/dev/null || true) | awk '{sub(":","",$2); print $2}' | paste -sd, -)}
IFS=, read -ra G <<< "${GPUS:-0}"; NGPU=${#G[@]}
SERVER_PID=
start_server() {   # once per invocation, on a node-local unix socket
    [ -n "$SERVER_PID" ] && return 0
    local scvd=${G[0]} dm=cuda:0
    if [ "$ARM" = m27 ] && [ "$NGPU" -gt 1 ]; then
        scvd=$GPUS; dm=sequential; export WORLDMM_MAX_MEMORY=${WORLDMM_MAX_MEMORY:-0:40GiB,1:28GiB}
    fi
    export WORLDMM_SERVER=/tmp/worldmm_$$.sock WORLDMM_AUTHKEY=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
    export SERVER_TRANSFORMERS=$("$SERVER_PY" -c "import transformers; print(transformers.__version__)")
    local ready=/tmp/worldmm_$$.ready slog=$WORK/logs/server_${ARM}_${V}_$(date +%m%d%H%M%S).log
    rm -f "$ready"
    CUDA_VISIBLE_DEVICES=$scvd WORLDMM_DEVICE_MAP=$dm "$SERVER_PY" -u "$M/epic/llm_server.py" --model_path "$MP" \
        --address "$WORLDMM_SERVER" --max_batch "${MAX_BATCH:-8}" --ready_file "$ready" > "$slog" 2>&1 &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null; rm -f $WORLDMM_SERVER /tmp/worldmm_$$.ready' EXIT
    echo "backbone server $MP on GPU(s) $scvd, log $slog"
    for _ in $(seq 1 180); do
        [ -f "$ready" ] && return 0
        kill -0 $SERVER_PID 2>/dev/null || { echo "server died:"; tail -20 "$slog"; exit 1; }
        sleep 5
    done
    echo "server not ready after 15 min"; exit 1
}
client() { CUDA_VISIBLE_DEVICES=${G[$((NGPU - 1))]} "$PY" -u "$@"; }

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4 (v3/DATA.md)"; exit 1; }
    "$PY" "$M/epic/prepare_epic.py" video --video "$VIDEOS/$P/$V.mp4" --subject "$V" --video_root "$WORK/videos_512" \
        --skeleton_root "$WORK/skeleton" --threads "${THREADS:-8}"
    record prep $((SECONDS - t))
}
build() {
    log build; local t=$SECONDS
    have_memory && { echo "memory complete in $OUT"; [ -s "$SKEL" ] || prep; return 0; }
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local build is started (a half-built local memory is resumed, never overwritten)
    if [ ! -d "$BUILD/captions/$V" ] && [ "${WORLDMM_FETCH:-1}" != 0 ]; then
        local rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline worldmm --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) # answers and HippoRAG caches made from an earlier local memory are stale
               rm -rf "$WORK/cache/$ARM/$V" "$WORK/rows/$ARM/$V"; rm -f "$OUT"/answers_*.json.gz "$OUT"/blind_*.json
               [ -s "$SKEL" ] || prep; record "build (fetched)" $((SECONDS - t)); return 0 ;;
            3) echo "no memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; WORLDMM_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    [ -s "$SKEL" ] && [ -s "$SMALL" ] || prep
    start_server
    mkdir -p "$BUILD/captions/$V"
    client "$M/epic/caption_epic.py" --skeleton "$SKEL" --output "$BUILD/captions/$V/10sec.json" --model "$MODEL"
    client "$M/epic/build_epic.py" --video "$V" --caption_root "$BUILD/captions" --meta_root "$BUILD/meta" \
        --model "$MODEL" --out "$OUT"
    have_memory || { echo "memory for $V is incomplete in $OUT"; exit 1; }
    record build $((SECONDS - t))
}
qa_file() {   # this bank's questions for the video, in WorldMM's format (asked at t_ask, or at the end)
    local bank=${BANK:?qa and blind need a bank}; QA=$WORK/qa/$bank/$V.json
    if [ ! -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ]; then   # banks are generated, not committed
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
    [ -s "$SKEL" ] || prep
    "$PY" "$M/epic/prepare_epic.py" questions --bank "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" \
        --skeleton "$SKEL" --out "$QA"
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS ans=$OUT/answers_$BANK$SUF.json.gz
    [ -s "$ans" ] && { echo "done: $ans"; return 0; }
    have_memory || build
    [ -s "$SMALL" ] || prep
    qa_file; start_server
    # a question that fails (WorldMM's "Error" / "Unable to generate answer") is retried by the next pass
    for k in 1 2 3; do
        client "$M/epic/eval_epic.py" --video "$V" --qa_file "$QA" --memory_dir "$OUT" --video_file "$SMALL" \
            --model "$MODEL" --rows "$WORK/rows/$ARM/$V/$BANK$SUF.jsonl" --out "$ans" \
            --cache_dir "$WORK/cache/$ARM/$V/$BANK$SUF" && break || echo "pass $k left questions unanswered"
    done
    [ -s "$ans" ] || { echo "still unanswered after 3 passes (rerun the same command to retry)"; exit 1; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; local t=$SECONDS
    [ -s "$OUT/blind_$BANK.json" ] && { echo "done: $OUT/blind_$BANK.json"; return 0; }
    qa_file; start_server
    client "$M/epic/eval_epic.py" --blind --video "$V" --qa_file "$QA" --model "$MODEL" \
        --rows "$WORK/rows/$ARM/$V/blind_$BANK.jsonl" --out "$OUT/blind_$BANK.json"
    record "blind $BANK" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline worldmm --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
