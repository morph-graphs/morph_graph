#!/usr/bin/env bash
# EgoRAG (EgoLife's EgoButler: EgoGPT captions + EgoRAG) on one HD-EPIC video, as released. See README.md.
#
#   bash baselines/egorag/run_egorag.sh <stage> <video> <arm> [bank]
#
#   stage  prep     30 s clips with audio (448 px) + the clip list                        CPU, ffmpeg
#          build    the question-independent memory: an EgoGPT caption per clip (one 48 GB GPU, local weights),
#                   the Chroma index (OpenCLIP text encoder), then EgoRAG's 10 min / 1 h / day summaries, which
#                   are PAID API calls (GPT-4o, DeepSeek). Without the keys build stops after the index
#                   (exit 4) and says which key is missing; nothing is called.
#          qa       EgoRAG answers every question of <bank> for this video: PAID API calls (GPT-4o), no GPU.
#                   Exit 4 without the key.
#          upload   this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all      prep + build, then qa when a bank is given (no upload)
#   arm    egogpt_gpt4o   the paper's configuration: captions by EgoGPT-7b (EgoIT-99K + EgoLife day 1),
#                         summaries, retrieval reasoning and answers by GPT-4o, hour summaries by DeepSeek
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
# No blind stage: the no-memory floor would be GPT-4o on question and options alone, which is not this method.
#
# Environment (see README.md):
#   PY               python of the egorag environment (env/setup_env.sh; the Docker image sets it)
#   CUDA_HOME        a CUDA 12 toolkit with nvcc, for the captions: deepspeed (imported by EgoGPT's Whisper
#                    encoder) checks it at import on a GPU machine (the image has it)
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   OPENAI_API_KEY   GPT-4o, for the summaries and qa      } from the environment only: never on a command
#   DEEPSEEK_API_KEY DeepSeek, for the hour summaries      } line, in a file or in a log
#                    (upstream's Azure variables take precedence when set: AZURE_API_KEY + AZURE_API_ENDPOINT,
#                    AZURE_DEEPSEEK_KEY + AZURE_DEEPSEEK_ENDPOINT [+ DEPLOYMENT_NAME])
#   EGORAG_GPT_MODEL       default gpt-4o             } the ids asked for; runs.jsonl records the ids the
#   EGORAG_DEEPSEEK_MODEL  default deepseek-v4-pro    } API returned (README.md: DeepSeek no longer lists R1)
#   EGORAG_EGOGPT    default lmms-lab/EgoGPT-7b-EgoIT-EgoLife (Hugging Face cache), or a local copy
#   EGORAG_WORK      default baselines/egorag/work: clips, the Chroma index, upstream's events_diary
#   EGORAG_OUT       default baselines/egorag/results: per <arm>/<video> the memory, answers_<bank>.json.gz,
#                    api_usage.jsonl (one line per paid call: model id returned, tokens) and runs.jsonl
#                    (`upload` puts them in the dataset morph-graphs/epic-morph-results at egorag/<arm>/<video>/)
#   EGORAG_TAG       a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   EGORAG_LIMIT     qa: answer at most this many unanswered questions (a smoke test)
#   EGORAG_FETCH     0 to never take the memory from the dataset (build normally reuses it when it is there)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_egorag.sh <prep|build|qa|upload|all> <video> <egogpt_gpt4o> [bank]}
V=${2:?video id}; ARM=${3:?arm egogpt_gpt4o}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/EgoLife   # R: the submodule (upstream)
PY=${PY:?set PY to the python of the egorag environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/EgoRAG/egorag/agents/RagAgent.py" ] || { echo "submodule missing: git submodule update --init baselines/egorag/EgoLife"; exit 1; }
[ "$ARM" = egogpt_gpt4o ] || { echo "arm must be egogpt_gpt4o"; exit 1; }
EGOGPT=${EGORAG_EGOGPT:-lmms-lab/EgoGPT-7b-EgoIT-EgoLife}
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${EGORAG_WORK:-$M/work}; OUTROOT=${EGORAG_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; CLIPS=$WORK/clips_448; SKEL=$WORK/prep/${V}_30sec_skeleton.json; CAP=$OUT/${V}_30sec.json
IDX=$WORK/index/${ARM}_$V; TAG=${EGORAG_TAG:-}; SUF=${TAG:+_$TAG}
mkdir -p "$OUT" "$WORK/locks"
# EgoLife.patch (the plain-key API branches of egorag/utils/util.py), applied once to the submodule's files
( flock 9
  cd "$R"
  if ! patch -p1 -R -s -f --dry-run < "$M/EgoLife.patch" >/dev/null 2>&1; then
      patch -p1 -s < "$M/EgoLife.patch" || { echo "EgoLife.patch does not apply to $R (is the submodule at the pinned commit?)"; exit 1; }
      echo "applied EgoLife.patch to $R"
  fi
) 9>"$WORK/locks/patch.lock"
export PYTHONPATH="$R/EgoRAG:$R/EgoGPT${PYTHONPATH:+:$PYTHONPATH}"
export EGORAG_GPT_MODEL=${EGORAG_GPT_MODEL:-gpt-4o} EGORAG_DEEPSEEK_MODEL=${EGORAG_DEEPSEEK_MODEL:-deepseek-v4-pro}
# Chroma's telemetry off: nothing leaves the machine but the API calls
export EGORAG_USAGE_LOG=$OUT/api_usage.jsonl ANONYMIZED_TELEMETRY=False
read -ra MEMORY <<< "$("$PY" "$M/epic/build_memory.py" --memory_files)"
have_memory() { local f; for f in "${MEMORY[@]}"; do [ -s "$OUT/${f//\{video\}/$V}" ] || return 1; done; }
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage, for the hand-back: record <stage> <seconds> <epoch the stage started>
    "$PY" - "$1" "$2" "$3" >> "$OUT/runs.jsonl" <<'EOF'
import hashlib, json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("EGORAG_REPO_COMMIT") or git("rev-parse", "HEAD")
up = os.environ.get("EGORAG_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/egorag/EgoLife")
bank = os.environ.get("EGORAG_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
# the submodule's files carry EgoLife.patch, so it is left out of "dirty" and the patch is named by its hash
dirty = bool(git("status", "--porcelain", "--ignore-submodules", "baselines/egorag")) if not os.environ.get("EGORAG_REPO_COMMIT") else None
api = {}   # the paid calls of this stage: per call and model id the API returned, the calls and tokens
log = os.environ["EGORAG_USAGE_LOG"]
for u in (json.loads(l) for l in open(log)) if os.path.exists(log) else []:
    if u["t"] >= float(sys.argv[3]):
        a = api.setdefault(f"{u['call']}:{u['model']}", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        a["calls"] += 1
        a["prompt_tokens"] += u["prompt_tokens"] or 0
        a["completion_tokens"] += u["completion_tokens"] or 0
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "egolife_commit": up, "egolife_patch": hashlib.sha1(open("baselines/egorag/EgoLife.patch", "rb").read()).hexdigest()[:12],
                  "epic_morph_commit": bank, "captioner": os.environ.get("EGOGPT"),
                  "models_asked": {"gpt": os.environ["EGORAG_GPT_MODEL"], "deepseek": os.environ["EGORAG_DEEPSEEK_MODEL"]},
                  "api": api, "torch": torch.__version__, "transformers": transformers.__version__, "gpus": gpus,
                  "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
export EPIC EGOGPT
cd "$REPO"

prep() {
    log prep; local t=$SECONDS t0; t0=$(date +%s.%N); local P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4"; exit 1; }
    ( flock 9   # one prep per video at a time
      [ -s "$SKEL" ] || "$PY" "$M/epic/prepare_clips.py" --video "$VIDEOS/$P/$V.mp4" --subject "$V" \
          --clip_root "$CLIPS" --data_dir "$WORK/prep" --threads "${THREADS:-8}"
    ) 9>"$WORK/locks/prep_$V.lock"
    record prep $((SECONDS - t)) "$t0"
}
build() {
    log build; local t=$SECONDS t0 rc=0; t0=$(date +%s.%N)
    # the memory depends only on the video: take it from the dataset when it is there and no local one is started
    if [ ! -s "$CAP" ] && [ "${EGORAG_FETCH:-1}" != 0 ]; then
        "$PY" "$M/../common/hf_results.py" fetch-memory --baseline egorag --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -rf "$IDX"; rm -f "$OUT"/answers_*.json.gz ;;   # made from an earlier local memory
            3) echo "no EgoRAG memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; EGORAG_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if ! have_memory; then
        if ! "$PY" -c "import json,sys; sys.exit(not all(c.get('text') for c in json.load(open(sys.argv[1]))))" "$CAP" 2>/dev/null; then
            [ -s "$SKEL" ] || prep
            "$PY" "$M/epic/caption_clips_egogpt.py" --skeleton "$SKEL" --output "$CAP" --clip_root "$CLIPS" --model "$EGOGPT"
        fi
        "$PY" "$M/epic/build_memory.py" --video "$V" --captions "$CAP" --out "$OUT" --work "$IDX" --step db
        record "build captions+index" $((SECONDS - t)) "$t0"
        rc=0; "$PY" "$M/epic/build_memory.py" --video "$V" --captions "$CAP" --out "$OUT" --work "$IDX" --step events || rc=$?
        [ $rc != 4 ] || exit 4   # no key: nothing was called
        # the calls a failed run made are still put on record
        [ $rc = 0 ] || { record "build summaries (failed)" $((SECONDS - t)) "$t0"; exit $rc; }
    fi
    have_memory || { record "build (incomplete)" $((SECONDS - t)) "$t0"; echo "memory for $V is incomplete in $OUT"; exit 1; }
    record build $((SECONDS - t)) "$t0"
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
qa() {
    log "qa $BANK$SUF"; local t t0 rc=0
    have_memory || build
    t=$SECONDS; t0=$(date +%s.%N)   # after build: its calls and time are on its own line
    bank_file
    "$PY" "$M/epic/run_qa.py" --video "$V" --bank_file "$BF" --memory "$OUT" --work "$IDX" \
        --answers "$OUT/answers_$BANK$SUF.json.gz" ${EGORAG_LIMIT:+--limit "$EGORAG_LIMIT"} || rc=$?
    [ $rc != 4 ] || exit 4   # no key: nothing was called
    if [ $rc != 0 ]; then
        record "qa $BANK$SUF (incomplete)" $((SECONDS - t)) "$t0"
        echo "qa did not finish (exit $rc); rerun the same command to retry"; exit $rc
    fi
    record "qa $BANK$SUF" $((SECONDS - t)) "$t0"
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline egorag --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || qa ;;
    *) echo "stage must be prep, build, qa, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
