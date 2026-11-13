#!/usr/bin/env bash
# MERIT on one HD-EPIC video, without SLURM: the same steps and settings as the batch10 runs
# (MERIT 8B 36.3%, 27B 49.0% on the epic-morph bank batch10_agree2). See README.md here.
#
#   bash baselines/merit/run_merit.sh <stage> <video> <arm> [bank]
#
#   stage  prep   30 s clips (448 px) + 1 fps frames + clip list                 CPU, ffmpeg
#          build  MERIT's memory: a caption per clip, then its retrieval keys     GPU
#          qa     MERIT answers every question of <bank> for this video            GPU (needs build)
#          blind  the same answer prompt with no context: the no-memory floor      GPU (no build needed)
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    m8   Qwen3-VL-8B (captions and answers)    one GPU with >= 40 GB (8B + the 4B embedding model)
#          m27  Qwen3.8-27B (captions and answers)    one 80 GB GPU, or 2x 48 GB, or 4x 24 GB
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing; byte-identical for anyone at that commit)
#
# Environment (see README.md): PY  python of the arm's environment (the Docker image sets it)
#   EPIC_MORPH_DIR an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS  default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   MERIT_WORK     default baselines/merit/work: clips, frames, run folders, caches (not committed)
#   MERIT_OUT      default baselines/merit/results: per <arm>/<video> the memory (captions, keys),
#                  answers_<bank>.json.gz, blind_<bank>.json and runs.jsonl (not committed; `upload` puts
#                  them in the dataset morph-graphs/epic-morph-results at merit/<arm>/<video>/)
#   MERIT_TAG      a variant name for the answers (answers_<bank>_<tag>.json.gz); MERIT_NO_FRAMES=1 (answer from
#                  captions only, memory probes) defaults it to "noframes", so variants never overwrite each other
#   MERIT_CONTEXT  blind only: a JSON file {question ID: context text}; the model answers from that text instead of an
#                  empty context, with no search. Needs MERIT_TAG (one per context file); written to blind_<bank>_<tag>.json
#   MERIT_FETCH    0 to never take the memory from the dataset (build normally reuses it when it is there)
# Every step skips finished work, so rerunning the same command resumes. The memory is built once per video
# and arm and shared by every bank, probe set and variant.
set -euo pipefail
STAGE=${1:?usage: run_merit.sh <prep|build|qa|blind|upload|all> <video> <m8|m27> [bank]}
V=${2:?video id}; ARM=${3:?arm m8 or m27}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/MERIT   # R: the MERIT submodule
PY=${PY:?set PY to the python of the $ARM environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/eval/evaluate_egolife.py" ] || { echo "MERIT submodule missing: git submodule update --init baselines/merit/MERIT"; exit 1; }
case $ARM in m8) MODEL=qwen3vl-8b ;; m27) MODEL=qwen38-27b ;; *) echo "arm must be m8 or m27"; exit 1 ;; esac
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${MERIT_WORK:-$M/work}; OUTROOT=${MERIT_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; PREP=$WORK/prep; CAP=$OUT/${V}_30sec.json
TAG=${MERIT_TAG:-}; [ "${MERIT_NO_FRAMES:-0}" = 1 ] && [ -z "$TAG" ] && TAG=noframes; SUF=${TAG:+_$TAG}
mkdir -p "$OUT" "$WORK/logs"
# eval_egolife.sh calls plain `python`: this environment's first
export PATH="$(dirname "$PY"):$PATH" PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); GPU_IDS=$CUDA_VISIBLE_DEVICES
elif [ "$NGPU" -gt 1 ] && [ "$STAGE" != prep ]; then
    # MERIT would spread one model over every GPU it sees; choose them: one H100 for m27, one GPU for m8
    echo "several GPUs visible: set CUDA_VISIBLE_DEVICES to the GPU(s) for this job (e.g. CUDA_VISIBLE_DEVICES=0)"; exit 1
else
    GPU_IDS=$(seq -s, 0 $((NGPU - 1)))
fi
if [ "$NGPU" -gt 1 ]; then
    export MERIT_DEVICE_MAP=auto
    # a model near the GPUs' total size needs more of GPU 0, and answering split over GPUs hit an
    # illegal memory access without blocking launches (Qwen3.8-27B on 4x A5000, 2026-11-10)
    [ "$ARM" = m27 ] && export MERIT_GPU0_FRAC=0.6 CUDA_LAUNCH_BLOCKING=1
else
    export MERIT_DEVICE_MAP=cuda:0
fi
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'), $NGPU GPU)"; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# docker/run.sh passes the commits read on the host (a worktree's .git points outside the container)
commit = os.environ.get("MERIT_REPO_COMMIT") or git("rev-parse", "HEAD")
merit = os.environ.get("MERIT_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/merit/MERIT")
bank = os.environ.get("MERIT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/merit"))
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "merit_commit": merit, "epic_morph_commit": bank,
                  "model": os.environ.get("MODEL"), "torch": torch.__version__, "transformers": transformers.__version__,
                  "gpus": gpus, "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
export MODEL EPIC
cd "$REPO"

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4 (v3/DATA.md)"; exit 1; }
    "$PY" "$M/epic/prepare_epic.py" --video "$VIDEOS/$P/$V.mp4" --subject "$V" --clip_root "$WORK/clips_448" \
        --frame_root "$WORK/frames_448" --data_dir "$PREP" --clip_size 448 --frame_size 448 --threads "${THREADS:-8}"
    record prep $((SECONDS - t))
}
build() {
    log build; local t=$SECONDS
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local one is started (a half-built local memory is resumed, never overwritten)
    if [ ! -s "$CAP" ] && [ ! -s "$OUT/${V}_30sec_4_key.json" ] && [ "${MERIT_FETCH:-1}" != 0 ]; then
        local rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) # anything made from an earlier local memory is stale: embeddings, answering runs, answers
               rm -rf "$WORK/cache/${ARM}_$V" "$WORK/runs/$ARM/$V"; rm -f "$OUT"/answers_*.json.gz ;;
            3) echo "no memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; MERIT_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    [ -s "$PREP/captions/$V/${V}_30sec_skeleton.json" ] || prep
    "$PY" "$M/epic/caption_clips_qwen.py" --model "$MODEL" --skeleton "$PREP/captions/$V/${V}_30sec_skeleton.json" --output "$CAP"
    # extract_keys resumes only the clips already in its own output: redo it if that output covers fewer
    # clips than the (now complete) captions, e.g. after an interrupted build
    local keys=$OUT/${V}_30sec_4_key.json
    if [ -s "$keys" ] && ! "$PY" -c "import json,sys; sys.exit(len(json.load(open(sys.argv[1]))) != len(json.load(open(sys.argv[2]))))" "$CAP" "$keys"; then
        mv "$keys" "$keys.partial"; rm -rf "$WORK/cache/${ARM}_$V"     # its embeddings were built from the old keys
    fi
    "$PY" "$R/preprocess/extract_keys.py" --input-file "$CAP" --model "$MODEL" --concurrency 1   # resumes; no-op when done
    "$PY" -c "import json,sys; k=json.load(open(sys.argv[1])); sys.exit(not k or any(not x.get('event_key') for x in k))" "$keys" \
        || { echo "keys for $V are missing or incomplete: $keys"; exit 1; }
    record build $((SECONDS - t))
}
qa_file() {   # this bank's questions for the video, in MERIT's format (asked at t_ask, or at the end)
    local bank=${BANK:?qa and blind need a bank}; QA=$WORK/qa/$bank/EgoLifeQA_$V.json
    if [ ! -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ]; then   # banks are generated, not committed
        # one generator at a time: jobs of other videos read this checkout's bank files too
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
    [ -s "$PREP/captions/$V/${V}_30sec_skeleton.json" ] || prep
    mkdir -p "$(dirname "$QA")"
    "$PY" - "$EPIC/outputs/v3/banks/$bank/${V}_questions.json" "$V" "$PREP/captions/$V/${V}_30sec_skeleton.json" "$QA" <<'EOF'
import json, sys
sys.path.insert(0, "baselines/merit/epic")
from prepare_epic import convert_questions
bank, v, skel, out = sys.argv[1:]
end = json.load(open(skel))[-1]["end_time"]                      # HHMMSS00
end_s = int(end[:2]) * 3600 + int(end[2:4]) * 60 + int(end[4:6])
json.dump(convert_questions(bank, v, end_s), open(out, "w"), indent=2)
EOF
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS runs=$WORK/runs/$ARM/$V/$BANK$SUF sel
    [ -s "$OUT/${V}_30sec_4_key.json" ] || build
    qa_file; mkdir -p "$runs"
    local n; n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$QA")
    # Question IDs are the bank's keys, and an answer is reused only for an unchanged question (collect_answers.py),
    # so the answers file of an earlier tranche (or the dataset's) is a source like any run. Pass "full" answers
    # every question (MERIT resumes it from its temp file if interrupted); later passes, each under a new name so
    # no earlier answers are overwritten, answer only the IDs with no valid answer yet: new or edited questions,
    # or ones MERIT failed on (it writes "Error"). Nothing runs when all are answered.
    local ans=$OUT/answers_$BANK$SUF.json.gz
    collect() { "$PY" "$M/epic/collect_answers.py" --qa_file "$QA" --runs "$runs/*/*_final.json" --prev "$ans" --out "$ans" | tail -1; }
    for k in 1 2 3; do
        if [ -s "$ans" ] || ls "$runs"/*/*_final.json >/dev/null 2>&1; then
            missing=$(collect)
            [ -z "$missing" ] && break
            sel=(--question-ids $missing); name=${ARM}_${V}_${BANK}${SUF}_fill$(date +%m%d%H%M%S)
        else
            sel=(--start-num 1 --end-num "$n"); name=${ARM}_${V}_${BANK}${SUF}_full
        fi
        bash "$R/script/eval_egolife.sh" --subject "$V" --data-dir "$PREP" --video-root "$WORK/clips_448" \
            --exp-name "$name" --respond-model "$MODEL" --top-k 5 --max-rounds 5 \
            --image-max-tokens 512 --num-workers 1 --gpu-ids "$GPU_IDS" --gpus-per-worker "$NGPU" \
            --key event object summary --cache-dir "$WORK/cache/${ARM}_$V" --caption-dir "$OUT" --qa-file "$QA" \
            "${sel[@]}" --output-dir "$runs" --log-dir "$WORK/logs" 2>&1 | tee "$WORK/logs/$name.log" \
            || echo "pass $name exited with an error; the next pass resumes or retries"
    done
    missing=$(collect)
    [ -z "$missing" ] || { echo "still unanswered after 3 passes: $missing (rerun the same command to retry)"; exit 1; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    local ctx=${MERIT_CONTEXT:-} suf=
    # a tag per context file: two files sharing one output would reuse each other's answers on resume
    [ -z "$ctx" ] || suf=_${MERIT_TAG:?MERIT_CONTEXT needs MERIT_TAG, a name for this context condition}
    [ -z "$ctx" ] || [ -s "$ctx" ] || { echo "MERIT_CONTEXT file missing: $ctx"; exit 1; }
    log "blind $BANK$suf"; local t=$SECONDS
    qa_file
    "$PY" "$M/epic/blind_qa.py" --model "$MODEL" --qa_file "$QA" --output "$OUT/blind_$BANK$suf.json" ${ctx:+--context_file "$ctx"}
    record "blind $BANK$suf" $((SECONDS - t))
}

upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    [ -s "$CAP" ] && [ -s "$OUT/${V}_30sec_4_key.json" ] || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
