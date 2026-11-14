#!/usr/bin/env bash
# Ego-R1 on one HD-EPIC video, without SLURM, with the models it was published with. See README.md.
#
#   bash baselines/egor1/run_egor1.sh <stage> <video> <arm> [bank]
#
#   stage  prep       30 s clips with audio, in the EgoLife layout upstream's tools read     CPU, ffmpeg
#          build      the question-independent memory: Gemini 2.5 Pro captions of the clips,
#                     GPT-4.1 summaries (10 min, 1 hour, 1 day)                              PAID APIs, no GPU
#          qa         the agent answers every question of <bank> for this video              GPU (8 GB) + PAID APIs
#          upload     this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all        prep + build, then qa when a bank is given (no upload)
#          firstturn  (a check, not a task) only the agent's first turn per question and the parsed tool
#                     call: the part of qa that needs no API                                 GPU (8 GB)
#   arm    gemini25pro  the paper's main configuration, with Gemini 2.5 Pro where the paper has the retired
#                       Gemini 1.5 Pro (the 30 s captions and video_llm)
#          llava        the paper's tool ablation (Table 4): video_llm = LLaVA-Video-7B, local (EGOR1_LLAVA_PY)
#          Both: agent Ego-R1-Agent-3B (local), rag and vlm GPT-4o, final answer GPT-4.1, the same memory.
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
#
# A stage that needs a paid API stops with exit code 5, before calling anything, when its key is not set:
#   GEMINI_API_KEY                                Gemini 2.5 Pro
#   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY   an Azure OpenAI resource with deployments gpt-4o and gpt-4.1, or
#   OPENAI_API_KEY                                a plain OpenAI key
# Set them in the environment only (never on a command line or in a file of this repo).
# Environment (see README.md):
#   PY               python of the environment (env/setup_env.sh)
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   EGOR1_AGENT      a local copy of Ego-R1/Ego-R1-Agent-3B; default the Hugging Face id (cache)
#   EGOR1_LLAVA_PY   arm llava: python of an environment with LLaVA-NeXT, for upstream's llava_video.py server
#   EGOR1_WORK       default baselines/egor1/work: clips, tool caches, logs
#   EGOR1_OUT        default baselines/egor1/results: per <arm>/<video> the memory, cutoffs/ (the memory as seen at
#                    each question time), answers_<bank>.json.gz, api_usage.jsonl and runs.jsonl
#   EGOR1_TAG        a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   EGOR1_FETCH      0 to never take the memory from the dataset (build normally reuses it when it is there)
#   EGOR1_LIMIT      qa / firstturn: only the first N unanswered questions (a smoke test)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_egor1.sh <prep|build|qa|firstturn|upload|all> <video> <gemini25pro|llava> [bank]}
V=${2:?video id}; ARM=${3:?arm gemini25pro or llava}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/Ego-R1   # R: the upstream checkout
PY=${PY:?set PY to the python of the environment (env/setup_env.sh)}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/api/rag/r1rag/ragagent.py" ] || { echo "run: git submodule update --init baselines/egor1/Ego-R1"; exit 1; }
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${EGOR1_WORK:-$M/work}; OUTROOT=${EGOR1_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; CLIPS=$WORK/clips; VIDEO=$VIDEOS/${V%%-*}/$V.mp4
TAG=${EGOR1_TAG:-}; SUF=${TAG:+_$TAG}
case $ARM in gemini25pro|llava) ;; *) echo "arm must be gemini25pro or llava"; exit 1 ;; esac
mkdir -p "$OUT" "$WORK/logs" "$WORK/locks"
# our three changes to upstream (epic-morph.patch), applied once to the submodule's files; one task at a time
patched() { grep -q EGOR1_GEMINI_MODEL "$R/api/visual_tools/egor1_videollm_gemini/video_llm.py"; }
( flock 9; patched || patch -d "$R" -p1 -s --forward --no-backup-if-mismatch < "$M/epic-morph.patch" ) 9>"$WORK/locks/patch.lock"
patched || { echo "could not apply $M/epic-morph.patch to $R (is the submodule at its pinned commit?)"; exit 1; }
export EGOR1_SRC=$R
read -ra MEMORY <<< "$("$PY" "$M/epic/build_memory.py" --memory_files)"
have_memory() { local f; for f in "${MEMORY[@]}"; do [ -s "$OUT/${f//\{video\}/$V}" ] || return 1; done; }
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
USAGE=$OUT/api_usage.jsonl
mark() { U0=$( [ -f "$USAGE" ] && wc -c < "$USAGE" || echo 0); }   # where this stage's API calls start in the log
record() {   # one line per finished stage, for the hand-back: commits, models, and what the APIs were asked
    "$PY" - "$1" "$2" "$USAGE" "${U0:-0}" >> "$OUT/runs.jsonl" <<'PYEOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
sys.path.insert(0, "baselines/egor1/epic")
import apis
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("EGOR1_REPO_COMMIT") or git("rev-parse", "HEAD")
up = os.environ.get("EGOR1_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/egor1/Ego-R1")
bank = os.environ.get("EGOR1_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
models = dict(apis.MODELS, **({"video_llm": "lmms-lab/LLaVA-Video-7B-Qwen2"} if os.environ["ARM"] == "llava" else {}))
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "egor1_commit": up,
                  "epic_morph_commit": bank, "models": models,
                  # per "<step> <model id the API answered with>": calls and tokens (thinking apart) of this stage
                  "api_usage": apis.usage_summary(sys.argv[3], int(sys.argv[4])), "torch": torch.__version__,
                  "transformers": transformers.__version__, "gpus": gpus, "host": platform.node(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
PYEOF
}
export EPIC ARM
cd "$REPO"

prep() {
    log prep; local t=$SECONDS
    [ -s "$VIDEO" ] || { echo "missing $VIDEO"; exit 1; }
    ( flock 9; "$PY" "$M/epic/prepare_clips.py" --video "$VIDEO" --subject "$V" --clip_root "$CLIPS" --threads "${THREADS:-8}"
    ) 9>"$WORK/locks/prep_$V.lock"
    record prep $((SECONDS - t))
}
need_clips() { [ -s "$CLIPS/$V/A1_JAKE/DAY1/_clips.txt" ] || prep; }
build() {
    log build; local t=$SECONDS rc; mark
    # the memory depends only on the video: take it from the dataset when it is there and no local one is started
    if [ ! -s "$OUT/${V}_30sec.json" ] && [ ! -s "$OUT/${V}_30sec.json.partial" ] && [ "${EGOR1_FETCH:-1}" != 0 ]; then
        rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline egor1 --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -rf "$OUT/cutoffs"; rm -f "$OUT"/answers_*.json.gz ;;   # made from an earlier local memory
            3) echo "no Ego-R1 memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; EGOR1_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if ! have_memory; then
        need_clips
        "$PY" "$M/epic/build_memory.py" --video "$V" --clips "$CLIPS/$V" --out "$OUT"
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
LPID=
start_llava() {   # arm llava: upstream's LLaVA-Video-7B server (port 8000 is fixed in upstream's video_llm.py)
    [ "$ARM" = llava ] && [ -z "$LPID" ] || return 0
    local lpy=${EGOR1_LLAVA_PY:?arm llava: set EGOR1_LLAVA_PY to the python of the LLaVA-NeXT environment} i
    local llog=$WORK/logs/llava_$V.log
    (cd "$R/api/visual_tools/egor1_videollm_llava" && exec "$lpy" llava_video.py) > "$llog" 2>&1 &
    LPID=$!
    trap 'kill $LPID 2>/dev/null || true' EXIT
    for i in $(seq 1 120); do
        curl -s -o /dev/null "http://localhost:8000/docs" && break
        kill -0 $LPID 2>/dev/null || { echo "LLaVA-Video server died:"; tail -30 "$llog"; exit 1; }
        sleep 5
    done
    curl -s -o /dev/null "http://localhost:8000/docs" || { echo "LLaVA-Video server not up after 10 min"; exit 1; }
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS k rc=0
    have_memory || build
    mark
    need_clips; bank_file; start_llava
    # each pass answers what has no valid answer yet (a failed API call, no readable letter), with a new sampling seed
    for k in 1 2 3; do
        rc=0; "$PY" "$M/epic/run_qa.py" --video "$V" --arm "$ARM" --bank_file "$BF" --memory "$OUT" --clips "$CLIPS/$V" \
            --answers "$OUT/answers_$BANK$SUF.json.gz" --cache_dir "$WORK/cache/${ARM}_$V" --attempt $((k - 1)) ${EGOR1_LIMIT:+--limit "$EGOR1_LIMIT"} || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "qa did not finish (exit $rc); rerun the same command to retry"; exit $rc; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
firstturn() {
    log "firstturn $BANK"; local t=$SECONDS
    bank_file; [ -n "${EGOR1_VIDEO_END:-}" ] || [ -s "$VIDEO" ] || { echo "missing $VIDEO"; exit 1; }
    # the video's length: ffprobe on the mp4, or EGOR1_VIDEO_END (seconds) where there is no ffprobe
    "$PY" "$M/epic/run_qa.py" --video "$V" --arm "$ARM" --bank_file "$BF" --clips "$CLIPS/$V" --first_turn \
        --video_file "$VIDEO" ${EGOR1_VIDEO_END:+--video_end "$EGOR1_VIDEO_END"} --answers "$OUT/answers_$BANK$SUF.json.gz" --cache_dir "$WORK/cache/${ARM}_$V" \
        ${EGOR1_LIMIT:+--limit "$EGOR1_LIMIT"}
    record "firstturn $BANK" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline egor1 --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; firstturn) firstturn ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || qa ;;
    *) echo "stage must be prep, build, qa, firstturn, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
