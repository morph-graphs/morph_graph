#!/usr/bin/env bash
# HippoMM on one HD-EPIC video, without SLURM, with the models of its paper and official code. See README.md.
#
#   bash baselines/hippomm/run_hippomm.sh <stage> <video> <arm> [bank]
#
#   stage  prep   upstream's frame extraction (adaptive, <= 1 frame/s, full resolution) and audio   CPU, ffmpeg
#          build  the question-independent memory: segments, ImageBind features, Whisper, key frames,
#                 one Qwen2.5-VL-7B caption per key frame, the summary                              GPU
#          count  how many gpt-4o calls and input tokens <bank>'s questions need (a range), and their
#                 cost at list price: no GPU, no API                                                (needs build)
#          qa     HippoMM answers every question of <bank> for this video, each from the memory up to
#                 its t_ask                                           GPU + OPENAI_API_KEY (gpt-4o) (needs build)
#          blind  upstream's final answer prompt with no memory: the no-memory floor   OPENAI_API_KEY (gpt-4o)
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    prep + build, then count when a bank is given (never qa or blind: they cost money)
#   arm    q25vl7b-gpt4o   the paper's configuration: Qwen2.5-VL-7B-Instruct (captions, summaries; local),
#                          ImageBind-Huge, Whisper distil-large-v3 (local), gpt-4o (reasoning; OpenAI API)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
#
# Environment (see README.md):
#   PY               python of the client environment (env/setup_env.sh client)
#   HIPPOMM_VLLM_PY  python of the vllm environment (env/setup_env.sh vllm); it serves Qwen2.5-VL-7B for the
#                    stage and is stopped at exit. Or HIPPOMM_VLM_URL: an OpenAI-compatible endpoint already
#                    serving Qwen/Qwen2.5-VL-7B-Instruct
#   HIPPOMM_IMAGEBIND  folder holding imagebind_huge.pth (else ImageBind downloads it into ./.checkpoints)
#   OPENAI_API_KEY   for qa and blind only; read from the environment, never written anywhere
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   HIPPOMM_WORK     default baselines/hippomm/work: upstream's memory_store (frames, audio), server logs
#   HIPPOMM_OUT      default baselines/hippomm/results: per <arm>/<video> the memory, summaries.json,
#                    count_<bank>.json, answers_<bank>.json.gz, blind_<bank>.json.gz and runs.jsonl
#   HIPPOMM_TAG      a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   HIPPOMM_FETCH    0 to never take the memory from the dataset (build normally reuses it when it is there)
#   HIPPOMM_LIMIT    qa: only the first N unanswered questions (smoke tests)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_hippomm.sh <prep|build|count|qa|blind|upload|all> <video> <arm> [bank]}
V=${2:?video id}; ARM=${3:?arm q25vl7b-gpt4o}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/HippoMM   # R: the upstream checkout
PY=${PY:?set PY to the python of the client environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/hippomm/core/hippocampal_memory.py" ] || { echo "upstream checkout missing: $R"; exit 1; }
[ "$ARM" = q25vl7b-gpt4o ] || { echo "arm must be q25vl7b-gpt4o (the paper's configuration)"; exit 1; }
VLM=Qwen/Qwen2.5-VL-7B-Instruct
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; VIDEO=$VIDEOS/${V%%-*}/$V.mp4
WORK=${HIPPOMM_WORK:-$M/work}; OUTROOT=${HIPPOMM_OUT:-$M/results}; OUT=$OUTROOT/$ARM/$V; STORE=$WORK/store
TAG=${HIPPOMM_TAG:-}; SUF=${TAG:+_$TAG}
mkdir -p "$OUT" "$WORK/logs" "$STORE"
export HIPPOMM_SRC=$R NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
# upstream writes fixed-name temporary frames and audio: one folder per run, so jobs on a node do not collide
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/hippomm_${V}_XXXXXX")
SPID=
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null; rm -rf "$TMPDIR"' EXIT
# ctranslate2 (Whisper) loads the cuDNN 8 and cuBLAS that the torch 2.1.2 wheel carries in torch/lib
export LD_LIBRARY_PATH=$("$PY" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
read -ra MEMORY <<< "$("$PY" "$M/epic/build_memory.py" --memory_files)"
have_memory() { local f; for f in "${MEMORY[@]}"; do [ -s "$OUT/${f//\{video\}/$V}" ] || return 1; done; }
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, faster_whisper
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("HIPPOMM_REPO_COMMIT") or git("rev-parse", "HEAD")
sub = os.environ.get("HIPPOMM_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/hippomm/HippoMM")
bank = os.environ.get("HIPPOMM_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "hippomm_commit": sub,
                  "epic_morph_commit": bank, "vlm": os.environ.get("VLM"), "vllm": os.environ.get("VLLM_VERSION"),
                  "reasoning": "gpt-4o", "torch": torch.__version__, "faster_whisper": faster_whisper.__version__,
                  "gpus": gpus, "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
export EPIC VLM
cd "$REPO"

start_vlm() {   # Qwen2.5-VL-7B-Instruct behind an OpenAI-compatible endpoint, as upstream serves it (vLLM)
    [ -n "${HIPPOMM_VLM_URL:-}" ] && return
    local vpy=${HIPPOMM_VLLM_PY:?set HIPPOMM_VLLM_PY (env/setup_env.sh vllm) or HIPPOMM_VLM_URL} port slog i
    port=${HIPPOMM_PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 20000))}
    slog=$WORK/logs/vllm_${V}_$port.log
    # ImageBind (~5 GB) and Whisper (~2 GB) share the GPU with it: vLLM gets a fraction (0.6 of 48 GB holds the
    # weights and a 128k-token context)
    "$vpy" -m vllm.entrypoints.openai.api_server --model "$VLM" --served-model-name "$VLM" --host 127.0.0.1 \
        --port "$port" --seed 0 --gpu-memory-utilization "${HIPPOMM_VLLM_MEM:-0.6}" > "$slog" 2>&1 &
    SPID=$!
    for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break
        kill -0 $SPID 2>/dev/null || { echo "vLLM died:"; tail -30 "$slog"; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "vLLM not up after 15 min"; tail -30 "$slog"; exit 1; }
    echo "vLLM up after $((i * 5)) s on port $port, log $slog"
    export HIPPOMM_VLM_URL=http://127.0.0.1:$port/v1
    export VLLM_VERSION=$("$vpy" -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1)
}

prep() {
    log prep; local t=$SECONDS
    [ -s "$VIDEO" ] || { echo "missing $VIDEO"; exit 1; }
    mkdir -p "$WORK/locks"
    ( flock 9   # one extraction per video at a time
      "$PY" "$M/epic/build_memory.py" --stage prep --video "$V" --video_file "$VIDEO" --store "$STORE"
    ) 9>"$WORK/locks/prep_$V.lock"
    record prep $((SECONDS - t))
}
build() {
    log build; local t=$SECONDS rc
    # the memory depends only on the video and the models: take it from the dataset when it is there
    if ! have_memory && [ "${HIPPOMM_FETCH:-1}" != 0 ]; then
        rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline hippomm --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -f "$OUT"/answers_*.json.gz "$OUT"/count_*.json "$OUT/summaries.json" ;;   # made from an earlier memory
            3) echo "no HippoMM memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; HIPPOMM_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if ! have_memory; then
        [ -s "$STORE/frames/$V/metadata.yaml" ] || prep
        start_vlm
        "$PY" "$M/epic/build_memory.py" --stage build --video "$V" --video_file "$VIDEO" --store "$STORE" --out "$OUT"
        rm -f "$STORE"/checkpoints/checkpoint_"${V}"_*.json   # upstream's segment checkpoint (large); the memory replaces it
    fi
    have_memory || { echo "memory for $V is incomplete in $OUT"; exit 1; }
    record build $((SECONDS - t))
}
bank_file() {   # this bank's questions for the video (banks are generated in epic-morph, not committed)
    local bank=${BANK:?this stage needs a bank}; BF=$EPIC/outputs/v3/banks/$bank/${V}_questions.json
    if [ ! -s "$BF" ]; then
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$BF" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$BF" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
}
count() {
    log "count $BANK"
    have_memory || build
    bank_file
    "$PY" "$M/epic/run_qa.py" --count --video "$V" --bank_file "$BF" --memory "$OUT" --answers "$OUT/count_$BANK.json"
}
need_key() { [ -n "${OPENAI_API_KEY:-}" ] || { echo "$STAGE calls gpt-4o (upstream's reasoning model): set OPENAI_API_KEY in the environment; \`count\` gives the calls and tokens first"; exit 1; }; }
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS rc=0
    need_key
    have_memory || build
    [ -s "$VIDEO" ] || { echo "missing $VIDEO"; exit 1; }
    bank_file; start_vlm
    "$PY" "$M/epic/run_qa.py" --video "$V" --bank_file "$BF" --memory "$OUT" --video_file "$VIDEO" --store "$STORE" \
        --answers "$OUT/answers_$BANK$SUF.json.gz" --limit "${HIPPOMM_LIMIT:-0}" || rc=$?
    [ $rc = 0 ] || { echo "qa did not finish (exit $rc); rerun the same command to retry"; exit 1; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; local t=$SECONDS
    need_key; bank_file
    "$PY" "$M/epic/run_qa.py" --blind --video "$V" --bank_file "$BF" --answers "$OUT/blind_$BANK.json.gz"
    record "blind $BANK" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline hippomm --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; count) count ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || count ;;
    *) echo "stage must be prep, build, count, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
