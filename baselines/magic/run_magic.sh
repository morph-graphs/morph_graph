#!/usr/bin/env bash
# MAGIC-Video on one HD-EPIC video, without SLURM: the same steps and settings as the batch10 runs. See README.md.
#
#   bash baselines/magic/run_magic.sh <stage> <video> <arm> [bank]
#
#   stage  prep   30 s clips (448 px) + 1 fps frames + clip list: MERIT's prep            CPU, ffmpeg
#          build  the question-independent memory: MERIT's 30 s captions (from the dataset, else built
#                 with MERIT), summaries, OpenIE, semantic memory, clip embeddings, memory graph  GPU
#          qa     narrative chains per question cutoff (cached), then MAGIC answers every question of
#                 <bank> for this video                                                  GPU (needs build)
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    prep + build, then qa when a bank is given (no upload)
# No blind stage: the no-memory floor depends only on the backbone, and MERIT's blind stage for the same arm
# (same model, same question rows) already measures it.
#   arm    m8   Qwen3-VL-8B-Instruct for every LLM role       one GPU with >= 40 GB, or 2 GPUs (server | MAGIC)
#          m27  Qwen3.8-27B (non-thinking) for every role     one 80 GB GPU, or 2x 48 GB / 3+ GPUs (server on all
#                                                             but the last, MAGIC's embedders on the last)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing)
#
# Environment (see README.md):
#   PY               python of the client environment (env/setup_env.sh client): MAGIC itself, and the m8 server
#   MAGIC_SERVER_PY  python of the m27 environment (env/setup_env.sh m27), for the m27 server; m8 uses PY
#   MERIT_PY         python for MERIT's captions when the dataset has none; default the server's
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   MAGIC_MODELS     a folder with local copies of the models (Qwen3-VL-8B-Instruct, Qwen3.8-27B, Qwen3-Embedding-4B,
#                    VLM2Vec-V2.0 with its Qwen2-VL-2B-Instruct base); default the Hugging Face ids (cache)
#   MAGIC_WORK       default baselines/magic/work: clips, frames, MERIT's captions, server logs, caches
#   MAGIC_OUT        default baselines/magic/results: per <arm>/<video> the memory, chains/ (per cutoff),
#                    answers_<bank>.json.gz and runs.jsonl (`upload` puts them in the
#                    dataset morph-graphs/epic-morph-results at magic/<arm>/<video>/)
#   MAGIC_TAG        a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   MAGIC_FETCH      0 to never take the memory from the dataset (build normally reuses it when it is there)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_magic.sh <prep|build|qa|upload|all> <video> <m8|m27> [bank]}
V=${2:?video id}; ARM=${3:?arm m8 or m27}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/MAGIC-video   # R: the submodule
PY=${PY:?set PY to the python of the client environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/src/worldmm/llm/openrouter.py" ] || { echo "submodule missing: git submodule update --init baselines/magic/MAGIC-video baselines/merit/MERIT"; exit 1; }
HUB=${MAGIC_MODELS:-}
hub() { if [ -n "$HUB" ]; then echo "$HUB/${1#*/}"; else echo "$1"; fi; }   # local copy, or the HF id
case $ARM in
    m8)  BACKBONE=$(hub Qwen/Qwen3-VL-8B-Instruct); SRV_PY=$PY ;;
    m27) BACKBONE=$(hub Qwen/Qwen3.8-27B); SRV_PY=${MAGIC_SERVER_PY:-} ;;
    *) echo "arm must be m8 or m27"; exit 1 ;;
esac
export MAGIC_TEXT_EMB=$(hub Qwen/Qwen3-Embedding-4B) MAGIC_VIS_EMB=$(hub VLM2Vec/VLM2Vec-V2.0) MAGIC_SRC=$R
WORK=${MAGIC_WORK:-$M/work}; OUTROOT=${MAGIC_OUT:-$M/results}; OUT=$OUTROOT/$ARM/$V
CLIPS=$WORK/merit/clips_448; MC=$WORK/merit/results/$ARM/$V          # MERIT's prep and captions
TAG=${MAGIC_TAG:-}; SUF=${TAG:+_$TAG}
mkdir -p "$OUT" "$WORK/logs"
export PYTHONPATH="$R/src:$R/src/HippoRAG/src:$M/epic${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
read -ra MEMORY <<< "$("$PY" "$M/epic/build_memory.py" --memory_files)"
have_memory() { local f; for f in "${MEMORY[@]}"; do [ -s "$OUT/${f//\{video\}/$V}" ] || return 1; done; }
merit() {   # MERIT's run script on this video, its files kept in the work folder
    MERIT_WORK=$WORK/merit MERIT_OUT=$WORK/merit/results MERIT_FETCH=0 bash "$M/../merit/run_merit.sh" "$@"
}
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("MAGIC_REPO_COMMIT") or git("rev-parse", "HEAD")
sub = os.environ.get("MAGIC_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/magic/MAGIC-video")
bank = os.environ.get("MAGIC_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/magic")) if not os.environ.get("MAGIC_REPO_COMMIT") else None
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "magic_commit": sub, "epic_morph_commit": bank, "model": os.environ.get("BACKBONE"),
                  "server_transformers": os.environ.get("SRV_TF"), "torch": torch.__version__,
                  "transformers": transformers.__version__, "gpus": gpus, "host": platform.node(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
export EPIC BACKBONE
cd "$REPO"

SPID=
start_server() {   # the arm's backbone behind an OpenAI-compatible endpoint; MAGIC sends every LLM call there
    [ -n "$SPID" ] && return
    [ -n "$SRV_PY" ] || { echo "set MAGIC_SERVER_PY to the python of the m27 environment"; exit 1; }
    local ids n g0 gmem budget slog
    ids=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}
    IFS=, read -ra G <<< "$ids"; n=${#G[@]}
    # GPU split: m8 on 2+ GPUs -> server on the first, MAGIC (embedders) on the last; m27 on 3+ GPUs -> server
    # on all but the last; otherwise both share every GPU
    SRV_GPUS=$ids; MAGIC_GPUS=$ids
    if [ "$ARM" = m8 ] && [ "$n" -ge 2 ]; then SRV_GPUS=${G[0]}; MAGIC_GPUS=${G[$((n - 1))]}; fi
    if [ "$ARM" = m27 ] && [ "$n" -ge 3 ]; then SRV_GPUS=$(IFS=,; echo "${G[*]:0:$((n - 1))}"); MAGIC_GPUS=${G[$((n - 1))]}; fi
    # KV-cache budget per server batch from the server's first GPU (24 GB -> 12k tokens, 48 GB -> 32k)
    g0=${SRV_GPUS%%,*}
    gmem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g0" | head -1)
    budget=64000; [ "$gmem" -lt 60000 ] && budget=32000; [ "$gmem" -lt 30000 ] && budget=12000
    budget=${TOKEN_BUDGET:-$budget}
    PORT=${MAGIC_PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 20000))}
    slog=$WORK/logs/server_${ARM}_${V}_$PORT.log
    echo "server GPUs $SRV_GPUS (${gmem} MiB, token budget $budget), MAGIC GPUs $MAGIC_GPUS, port $PORT, log $slog"
    CUDA_VISIBLE_DEVICES=$SRV_GPUS "$SRV_PY" "$M/epic/hf_server.py" --model "$BACKBONE" --port "$PORT" \
        --device_map "$( [ "$SRV_GPUS" = "$g0" ] && echo cuda:0 || echo auto)" --max_batch "${MAX_BATCH:-8}" \
        --token_budget "$budget" > "$slog" 2>&1 &
    SPID=$!
    trap 'kill $SPID 2>/dev/null || true' EXIT
    local i
    for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
        kill -0 $SPID 2>/dev/null || { echo "LLM server died:"; tail -30 "$slog"; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "LLM server not up after 15 min"; tail -30 "$slog"; exit 1; }
    echo "LLM server up after $((i * 5)) s: $(grep -m1 loaded "$slog")"
    export CUDA_VISIBLE_DEVICES=$MAGIC_GPUS MAGIC_LLM_BASE_URL=http://127.0.0.1:$PORT/v1 MAGIC_LLM_MODEL=local MAGIC_SEED=0
    export SRV_TF=$("$SRV_PY" -c "import transformers; print(transformers.__version__)")
}

prep() {
    log prep; local t=$SECONDS
    PY=$PY merit prep "$V" "$ARM"
    record prep $((SECONDS - t))
}
need_clips() { [ -s "$WORK/merit/prep/captions/$V/${V}_30sec_skeleton.json" ] || prep; }
build() {
    log build; local t=$SECONDS rc
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local one is started (a half-built local memory is resumed, never overwritten)
    if [ ! -s "$OUT/${V}_30sec.json" ] && [ "${MAGIC_FETCH:-1}" != 0 ]; then
        rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline magic --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) # anything made from an earlier local memory is stale: chains, index caches, answers
               rm -rf "$OUT/chains" "$WORK/cache/${ARM}_$V"; rm -f "$OUT"/answers_*.json.gz ;;
            3) echo "no MAGIC memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; MAGIC_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if ! have_memory; then
        need_clips
        # MERIT's captions for the arm: from the dataset (merit/<arm>/<video>), else built here with MERIT
        if [ ! -s "$OUT/${V}_30sec.json" ] && [ ! -s "$MC/${V}_30sec.json" ]; then
            rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline merit --arm "$ARM" --video "$V" --dir "$MC" || rc=$?
            case $rc in
                0) ;;
                3) echo "no MERIT memory for $V $ARM in the dataset: building MERIT's captions here"
                   PY=${MERIT_PY:-$SRV_PY} merit build "$V" "$ARM" ;;
                *) echo "could not reach the dataset for MERIT's captions"; exit 1 ;;
            esac
        fi
        start_server
        "$PY" "$M/epic/build_memory.py" --video "$V" --out "$OUT" --merit_captions "$MC/${V}_30sec.json" --clip_root "$CLIPS"
    fi
    have_memory || { echo "memory for $V is incomplete in $OUT"; exit 1; }
    record build $((SECONDS - t))
}
bank_file() {   # this bank's questions for the video (banks are generated in epic-morph, not committed)
    local bank=${BANK:?qa needs a bank}; BF=$EPIC/outputs/v3/banks/$bank/${V}_questions.json
    if [ ! -s "$BF" ]; then
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$BF" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$BF" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS k rc=0
    have_memory || build
    need_clips; bank_file; start_server   # a memory fetched from the dataset comes without the clips
    # the video's answers under its other banks (same arm, memory and variant): a question they answered is not asked again
    local earlier
    earlier=$("$PY" "$M/../common/hf_results.py" earlier-answers --baseline magic --memory "${MEMORY[@]}" --arm "$ARM" \
        --video "$V" --dir "$OUT" --epic "$EPIC" --suffix="$SUF" | grep -vxF "$OUT/answers_$BANK$SUF.json.gz" || true)
    # each pass answers what has no valid answer yet (MAGIC errors are retried); nothing runs when all are done
    for k in 1 2 3; do
        rc=0; "$PY" "$M/epic/run_qa.py" --video "$V" --bank_file "$BF" --memory "$OUT" --clip_root "$CLIPS" \
            --answers "$OUT/answers_$BANK$SUF.json.gz" --cache_dir "$WORK/cache/${ARM}_$V" --prev $earlier || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "qa did not finish (exit $rc); rerun the same command to retry"; exit 1; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline magic --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || qa ;;
    *) echo "stage must be prep, build, qa, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
