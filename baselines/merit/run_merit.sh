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
#          MERIT's published models (README "Reported-configuration arms"); gpt-* models are the OpenAI API, paid:
#          g5m      the memory, written by gpt-5-mini (prep, build, upload); shared by the three answer arms   no GPU
#          g5m-q8   answers from the g5m memory with Qwen3-VL-8B (qa, upload); no API call                    as m8
#          g5m-q27  answers from the g5m memory with Qwen3.8-27B (qa, upload); no API call                    as m27
#          g5m-g5   answers from the g5m memory with gpt-5 (qa, upload), top-k 10 as published      one GPU >= 16 GB
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
#   OPENAI_API_KEY g5m build and g5m-g5 qa: the key, from the environment only (never an argument or a file);
#                  every other arm and stage runs with it removed from its environment
#   MERIT_SERVICE_TIER  the OpenAI service tier, e.g. flex (half price, slower); default the API's standard tier
#   MERIT_API_CONCURRENCY  g5m build: captions / keys requested at once (default 8); MERIT_API_THREADS  g5m-g5 qa:
#                  questions answered at once (default 4)
#   MERIT_TOP_K    clips retrieved per search round (default 5; g5m-g5: 10); setting it defaults MERIT_TAG to top<k>
# Every step skips finished work, so rerunning the same command resumes. The memory is built once per video
# and arm and shared by every bank, probe set and variant.
set -euo pipefail
STAGE=${1:?usage: run_merit.sh <prep|build|qa|blind|upload|all> <video> <m8|m27|g5m|g5m-q8|g5m-q27|g5m-g5> [bank]}
V=${2:?video id}; ARM=${3:?arm m8, m27, g5m, g5m-q8, g5m-q27 or g5m-g5}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/MERIT   # R: the MERIT submodule
PY=${PY:?set PY to the python of the $ARM environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/eval/evaluate_egolife.py" ] || { echo "MERIT submodule missing: git submodule update --init baselines/merit/MERIT"; exit 1; }
# MODEL writes the memory (captions, keys); RESPOND decides the searches, filters the neighbours and answers
# (the same model for m8 and m27). MEM is the arm whose folder holds the memory: an answer arm (ARM != MEM)
# reads it from there and keeps only its answers in its own folder.
case $ARM in
    m8) MODEL=qwen3vl-8b ;; m27) MODEL=qwen38-27b ;;
    g5m) MODEL=gpt-5-mini ;;
    g5m-q8) MEM=g5m; MODEL=gpt-5-mini; RESPOND=qwen3vl-8b ;;
    g5m-q27) MEM=g5m; MODEL=gpt-5-mini; RESPOND=qwen38-27b ;;
    g5m-g5) MEM=g5m; MODEL=gpt-5-mini; RESPOND=gpt-5; TOPK=10 ;;   # top-k 10 with GPT-5, 5 with Qwen3-VL-8B (paper, Table 7)
    *) echo "arm must be m8, m27, g5m, g5m-q8, g5m-q27 or g5m-g5"; exit 1 ;;
esac
MEM=${MEM:-$ARM}; RESPOND=${RESPOND:-$MODEL}
case $ARM:$STAGE in
    g5m:qa|g5m:blind) echo "g5m is the memory only: answer from it with g5m-q8, g5m-q27 or g5m-g5"; exit 1 ;;
    g5m-*:blind) echo "blind uses no memory: it is the reasoning model's own (m8, m27; gpt-5: the API baseline)"; exit 1 ;;
esac
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${MERIT_WORK:-$M/work}; OUTROOT=${MERIT_OUT:-$M/results}
[ -n "${MERIT_WORK:-}" ] && umask 002   # a shared work folder: files the other workers' users can reuse and reset
OUT=$OUTROOT/$ARM/$V; MOUT=$OUTROOT/$MEM/$V; PREP=$WORK/prep; CAP=$MOUT/${V}_30sec.json; KEYS=$MOUT/${V}_30sec_4_key.json
TAG=${MERIT_TAG:-}; [ "${MERIT_NO_FRAMES:-0}" = 1 ] && [ -z "$TAG" ] && TAG=noframes
[ -n "${MERIT_TOP_K:-}" ] && [ -z "$TAG" ] && TAG=top$MERIT_TOP_K   # another top-k is a variant, never the arm's main answers
SUF=${TAG:+_$TAG}; TOPK=${MERIT_TOP_K:-${TOPK:-5}}
mkdir -p "$OUT" "$MOUT" "$WORK/logs"
# The OpenAI API: only a stage whose model is gpt-* keeps the key (g5m: writing the memory; g5m-g5: answering).
# Every call is counted in USAGE (epic/openai_usage.py) and goes into the stage's runs.jsonl line.
USAGE=$WORK/usage/$ARM/$V.jsonl; API=
case $ARM:$STAGE in g5m:build|g5m:all) API=$MODEL ;; g5m-g5:qa|g5m-g5:all) API=$RESPOND ;; esac
if [ -n "$API" ]; then
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "$ARM $STAGE calls the OpenAI API ($API): set OPENAI_API_KEY in the environment"; exit 1; }
    mkdir -p "$(dirname "$USAGE")"
    export MERIT_USAGE_LOG=$USAGE MERIT_OPENAI_CACHE=$WORK/openai_cache/${ARM}_$V   # the reply cache: a rerun pays nothing twice
else
    unset OPENAI_API_KEY MERIT_USAGE_LOG
fi
# eval_egolife.sh calls plain `python`: this environment's first
export PATH="$(dirname "$PY"):$PATH" PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .); GPU_IDS=$CUDA_VISIBLE_DEVICES
elif [ "$NGPU" -gt 1 ] && [ "$STAGE" != prep ] && [ "$STAGE" != upload ] && [ "$ARM" != g5m ]; then
    # MERIT would spread one model over every GPU it sees; choose them: one H100 for m27, one GPU for m8
    echo "several GPUs visible: set CUDA_VISIBLE_DEVICES to the GPU(s) for this job (e.g. CUDA_VISIBLE_DEVICES=0)"; exit 1
else
    GPU_IDS=$(seq -s, 0 $((NGPU - 1)))
fi
if [ "$NGPU" -gt 1 ]; then
    export MERIT_DEVICE_MAP=auto
    # a model near the GPUs' total size needs more of GPU 0, and answering split over GPUs hit an
    # illegal memory access without blocking launches (Qwen3.8-27B on 4x A5000, 2026-11-10)
    [ "$RESPOND" = qwen38-27b ] && export MERIT_GPU0_FRAC=0.6 CUDA_LAUNCH_BLOCKING=1
else
    export MERIT_DEVICE_MAP=cuda:0
fi
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'), $NGPU GPU)"; }
record() {   # one line per finished stage, for the hand-back
    local use=; case $1 in build*|qa*) use=${MERIT_USAGE_LOG:-} ;; esac   # the OpenAI calls since this stage's last line
    MERIT_USAGE_LOG=$use "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# docker/run.sh passes the commits read on the host (a worktree's .git points outside the container)
commit = os.environ.get("MERIT_REPO_COMMIT") or git("rev-parse", "HEAD")
merit = os.environ.get("MERIT_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/merit/MERIT")
bank = os.environ.get("MERIT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/merit"))
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
row = {"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
       "merit_commit": merit, "epic_morph_commit": bank,
       "model": os.environ.get("MODEL"), "torch": torch.__version__, "transformers": transformers.__version__,
       "gpus": gpus, "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
if os.environ["RESPOND"] != os.environ["MODEL"]:   # an answer arm: who answered, from which arm's memory
    row.update(respond_model=os.environ["RESPOND"], memory_arm=os.environ["MEM"], top_k=int(os.environ["TOPK"]))
if os.environ.get("MERIT_USAGE_LOG"):   # {role: {model id the API answered with: calls and billed tokens}}
    sys.path.insert(0, "baselines/merit/epic")
    from openai_usage import summary
    row["openai"] = summary(os.environ["MERIT_USAGE_LOG"])
print(json.dumps(row))
EOF
    # counted once: the next line of this stage (a rerun, another bank) starts from an empty log
    if [ -n "$use" ] && [ -s "$use" ]; then mv "$use" "$use.$(date +%Y%m%d%H%M%S)"; fi
}
export MODEL RESPOND MEM TOPK EPIC
cd "$REPO"

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4 (v3/DATA.md)"; exit 1; }
    # one prep per video at a time: two tasks of the same video (8B and 27B, on one machine) share these
    # folders, and cutting clips into the same folder at once broke one of them ("expected 35 segments, got 1")
    mkdir -p "$WORK/locks"
    (
        flock 9
        [ -s "$PREP/captions/$V/${V}_30sec_skeleton.json" ] && exit 0   # the other task made it
        "$PY" "$M/epic/prepare_epic.py" --video "$VIDEOS/$P/$V.mp4" --subject "$V" --clip_root "$WORK/clips_448" \
            --frame_root "$WORK/frames_448" --data_dir "$PREP" --clip_size 448 --frame_size 448 --threads "${THREADS:-8}"
    ) 9>"$WORK/locks/prep_$V.lock"
    record prep $((SECONDS - t))
}
build() {
    log build; local t=$SECONDS
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local one is started (a half-built local memory is resumed, never overwritten)
    if [ ! -s "$CAP" ] && [ ! -s "$KEYS" ] && [ "${MERIT_FETCH:-1}" != 0 ]; then
        local rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --arm "$MEM" --video "$V" --dir "$MOUT" || rc=$?
        case $rc in
            0) # anything made from an earlier local memory is stale: embeddings, answering runs, answers
               # (an answer arm compares the memory with the one its answers were made from: memory_ref)
               [ "$ARM" != "$MEM" ] || { rm -rf "$WORK/cache/${ARM}_$V" "$WORK/runs/$ARM/$V"; rm -f "$OUT"/answers_*.json.gz; } ;;
            3) echo "no memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; MERIT_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if [ "$ARM" != "$MEM" ]; then   # an answer arm never writes the memory: that is the memory arm's build (and its key)
        [ -s "$KEYS" ] && return 0
        echo "no finished $MEM memory of $V here or in the dataset: build it first (run_merit.sh build $V $MEM)"; exit 1
    fi
    [ -s "$PREP/captions/$V/${V}_30sec_skeleton.json" ] || prep
    local api=; case $MODEL in gpt-*) api=1 ;; esac
    "$PY" "$M/epic/caption_clips_qwen.py" --model "$MODEL" --skeleton "$PREP/captions/$V/${V}_30sec_skeleton.json" --output "$CAP" \
        ${api:+--concurrency "${MERIT_API_CONCURRENCY:-8}"}
    # extract_keys resumes only the clips already in its own output: redo it if that output covers fewer
    # clips than the (now complete) captions, e.g. after an interrupted build
    local keys=$KEYS
    if [ -s "$keys" ] && ! "$PY" -c "import json,sys; sys.exit(len(json.load(open(sys.argv[1]))) != len(json.load(open(sys.argv[2]))))" "$CAP" "$keys"; then
        mv "$keys" "$keys.partial"; rm -rf "$WORK/cache/${ARM}_$V"     # its embeddings were built from the old keys
    fi
    if [ -n "$api" ]; then   # the same script, started through epic/openai_usage.py, which counts its calls
        "$PY" "$M/epic/openai_usage.py" keys --input-file "$CAP" --model "$MODEL" --concurrency "${MERIT_API_CONCURRENCY:-8}"
    else
        "$PY" "$R/preprocess/extract_keys.py" --input-file "$CAP" --model "$MODEL" --concurrency 1   # resumes; no-op when done
    fi
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
import json, os, sys
sys.path.insert(0, "baselines/merit/epic")
from prepare_epic import convert_questions
bank, v, skel, out = sys.argv[1:]
end = json.load(open(skel))[-1]["end_time"]                      # HHMMSS00
end_s = int(end[:2]) * 3600 + int(end[2:4]) * 60 + int(end[4:6])
tmp = f"{out}.{os.getpid()}"   # then rename: an m8 and an m27 task of the video may write it at once (shared work)
json.dump(convert_questions(bank, v, end_s), open(tmp, "w"), indent=2)
os.replace(tmp, out)
EOF
}
memory_ref() {   # an answer arm's folder names the memory its answers were made from, by file hashes
    [ "$ARM" != "$MEM" ] || return 0
    local ref=$OUT/${V}_memory_ref.json new
    new=$("$PY" -c 'import hashlib, json, os, sys
print(json.dumps({"memory_arm": sys.argv[1], "sha256": {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()
                                                         for f in sys.argv[2:]}}, sort_keys=True))' "$MEM" "$CAP" "$KEYS")
    if [ -f "$ref" ] && [ "$(cat "$ref")" != "$new" ]; then
        echo "the $MEM memory of $V changed: removing this arm's answers made from the old one"
        rm -rf "$WORK/cache/${ARM}_$V" "$WORK/runs/$ARM/$V"; rm -f "$OUT"/answers_*.json.gz
    fi
    echo "$new" > "$ref"
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS runs=$WORK/runs/$ARM/$V/$BANK$SUF sel
    [ -s "$KEYS" ] || build
    memory_ref
    qa_file; mkdir -p "$runs"
    # a gpt-* reasoning model: MERIT's evaluation started through epic/openai_usage.py, which counts its calls
    local merit=(bash "$R/script/eval_egolife.sh")
    case $RESPOND in gpt-*) merit=("$PY" "$M/epic/openai_usage.py" eval --num-threads "${MERIT_API_THREADS:-4}") ;; esac
    local n; n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$QA")
    # Question IDs are the bank's keys, and an answer is reused only for an unchanged question (collect_answers.py),
    # so the answers file of an earlier tranche (or the dataset's) is a source like any run, and so are the video's
    # answers under its other banks (same arm, memory and variant): a question they answered is not asked again.
    # Pass "full" answers every question (MERIT resumes it from its temp file if interrupted); later passes, each
    # under a new name so no earlier answers are overwritten, answer only the IDs with no valid answer yet: new or edited questions,
    # or ones MERIT failed on (it writes "Error"). Nothing runs when all are answered.
    # An answer arm's folder holds its memory as {video}_memory_ref.json (memory_ref), the memory arm's the files.
    local ans=$OUT/answers_$BANK$SUF.json.gz earlier mem=("{video}_30sec.json" "{video}_30sec_4_key.json")
    [ "$ARM" = "$MEM" ] || mem=("{video}_memory_ref.json")
    earlier=$("$PY" "$M/../common/hf_results.py" earlier-answers --memory "${mem[@]}" --arm "$ARM" --video "$V" --dir "$OUT" \
        --epic "$EPIC" --suffix="$SUF" | grep -vxF "$ans" || true)
    collect() { "$PY" "$M/epic/collect_answers.py" --qa_file "$QA" --runs "$runs/*/*_final.json" --prev "$ans" $earlier --out "$ans" | tail -1; }
    for k in 1 2 3; do
        if [ -s "$ans" ] || [ -n "$earlier" ] || ls "$runs"/*/*_final.json >/dev/null 2>&1; then
            missing=$(collect)
            [ -z "$missing" ] && break
            sel=(--question-ids $missing); name=${ARM}_${V}_${BANK}${SUF}_fill$(date +%m%d%H%M%S)
        else
            sel=(--start-num 1 --end-num "$n"); name=${ARM}_${V}_${BANK}${SUF}_full
        fi
        "${merit[@]}" --subject "$V" --data-dir "$PREP" --video-root "$WORK/clips_448" \
            --exp-name "$name" --respond-model "$RESPOND" --top-k "$TOPK" --max-rounds 5 \
            --image-max-tokens 512 --num-workers 1 --gpu-ids "$GPU_IDS" --gpus-per-worker "$NGPU" \
            --key event object summary --cache-dir "$WORK/cache/${ARM}_$V" --caption-dir "$MOUT" --qa-file "$QA" \
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
    if [ "$ARM" != "$MEM" ]; then   # an answer arm: its answers, and the reference to the memory they were made from
        # (a changed reference mirrors the folder, so answers made from an older memory leave the dataset too)
        [ -s "$OUT/${V}_memory_ref.json" ] || { echo "no answers in $OUT: run qa first"; exit 1; }
        "$PY" "$M/../common/hf_results.py" upload --arm "$ARM" --video "$V" --dir "$OUT" --memory '{video}_memory_ref.json'
        return 0
    fi
    [ -s "$CAP" ] && [ -s "$KEYS" ] || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) case $ARM in
             g5m) prep; build ;;                                    # the memory only
             g5m-*) qa ;;                                           # answers only, from the g5m memory
             *) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
         esac ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
