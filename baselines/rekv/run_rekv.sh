#!/usr/bin/env bash
# ReKV (in-context video KV-cache retrieval) with Qwen3-VL on one HD-EPIC video, without SLURM, through the
# vendored port (vendor/MemStream). See README.md.
#
#   bash baselines/rekv/run_rekv.sh <stage> <video> <arm> [bank]
#
#   stage  prep   nothing to do (frames are decoded from the video); kept for the common interface
#          build  nothing to do: ReKV's memory is the KV-cache of the video, kept in host RAM while `qa` runs
#          qa     encode the video once (question-independent), then ReKV answers every question of <bank>,
#                 each retrieving only among the stored features that end by its t_ask              GPU
#          blind  the same answer prompt with no video: the no-memory floor                         GPU
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    qa + blind for the given bank (no upload)
#   arm    m8   Qwen/Qwen3-VL-8B-Instruct: 48 GB of GPU memory (one 48 GB GPU, or two 24 GB) and about 40 GB of
#               host RAM for a 30 min video
#          (m27, Qwen3.8-27B, is not ported: the ReKV code is written against Qwen3-VL's decoder)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
#
# Environment (see README.md):
#   PY               python of the environment (env/setup_env.sh, or the image: docker/run.sh); ffmpeg and ffprobe on PATH
#   EPIC_MORPH_DIR   an epic-morph checkout (required for qa and blind): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   REKV_MODELS      a folder with a local copy of the model (Qwen3-VL-8B-Instruct); default the Hugging Face id
#   REKV_OUT         default baselines/rekv/results: per <arm>/<video> memory.json (what was encoded: settings,
#                    blocks, sizes, times), answers_<bank>.json.gz, blind_<bank>.json.gz and runs.jsonl (`upload`
#                    puts them in the dataset morph-graphs/epic-morph-results at rekv/<arm>/<video>/)
#   REKV_TAG         a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   REKV_LIMIT       answer only the first N unanswered questions (smoke tests)
# Every step skips answered questions, so rerunning the same command resumes (the video is encoded again).
set -euo pipefail
STAGE=${1:?usage: run_rekv.sh <prep|build|qa|blind|upload|all> <video> <m8> [bank]}
V=${2:?video id}; ARM=${3:?arm m8}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd)
PY=${PY:?set PY to the python of the environment}
HUB=${REKV_MODELS:-}
case $ARM in
    m8) MODEL=Qwen/Qwen3-VL-8B-Instruct; [ -z "$HUB" ] || MODEL=$HUB/Qwen3-VL-8B-Instruct ;;
    *) echo "arm must be m8 (m27 is not ported, see README.md)"; exit 1 ;;
esac
OUTROOT=${REKV_OUT:-$M/results}; OUT=$OUTROOT/$ARM/$V
TAG=${REKV_TAG:-}; SUF=${TAG:+_$TAG}
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODEL
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
epic() { EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd); export EPIC; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'PYEOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("REKV_REPO_COMMIT") or git("rev-parse", "HEAD")
bank = os.environ.get("REKV_EPIC_COMMIT") or (git("rev-parse", "HEAD", cwd=os.environ["EPIC"]) if os.environ.get("EPIC") else None)
dirty = bool(git("status", "--porcelain", "baselines/rekv")) if not os.environ.get("REKV_REPO_COMMIT") else None
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "epic_morph_commit": bank, "model": os.environ.get("MODEL"), "torch": torch.__version__,
                  "transformers": transformers.__version__, "gpus": gpus, "host": platform.node(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
PYEOF
}
cd "$REPO"

video_file() {
    local root=${HDEPIC_VIDEOS:-}
    [ -n "$root" ] || { epic; root=$EPIC/data/HD-EPIC/videos; }
    VF=$root/${V%%-*}/$V.mp4
    [ -s "$VF" ] || { echo "no video $VF"; exit 1; }
}
build() { log "build: nothing to do (the memory is the KV-cache held in RAM during qa)"; }
bank_file() {   # this bank's questions for the video (banks are generated in epic-morph, not committed)
    local bank=${BANK:?qa and blind need a bank}; epic; BF=$EPIC/outputs/v3/banks/$bank/${V}_questions.json
    if [ ! -s "$BF" ]; then
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$BF" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$BF" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS
    bank_file; video_file
    "$PY" "$M/epic/run_memstream.py" --video "$V" --video_file "$VF" --bank_file "$BF" --model "$MODEL" \
        --answers "$OUT/answers_$BANK$SUF.json.gz" --memory_json "$OUT/memory.json" --limit "${REKV_LIMIT:-0}"
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK$SUF"; local t=$SECONDS
    bank_file
    "$PY" "$M/epic/run_memstream.py" --video "$V" --bank_file "$BF" --blind --model "$MODEL" \
        --answers "$OUT/blind_$BANK$SUF.json.gz" --limit "${REKV_LIMIT:-0}"
    record "blind $BANK$SUF" $((SECONDS - t))
}
upload() {   # the KV-cache is never stored: the folder is memory.json (what was encoded), answers and runs.jsonl
    log upload
    "$PY" "$M/../common/hf_results.py" upload --baseline rekv --memory none --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) log "prep: nothing to do" ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
