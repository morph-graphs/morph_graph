#!/usr/bin/env bash
# EGAgent on one HD-EPIC video, without SLURM, in the paper's configuration (its own models). See README.md.
#
#   bash baselines/egagent/run_egagent.sh <stage> <video> <arm> [bank]
#
#   stage  prep   1 fps frames of the video, at its own resolution                              CPU, ffmpeg
#          build  the memory, no questions needed: SigLIP 2 frame embeddings (GPU, local, free), then
#                 GPT-4.1 captions of every 30 s and the entity graph from them (PAID: OpenAI)
#          qa     EGAgent answers every question of <bank> for this video, each from the memory up to its
#                 t_ask (PAID: the arm's agent model; a GPU or CPU for SigLIP 2's text encoder)
#          blind  EGAgent's answer prompt with no retrieved context: the no-memory floor (PAID, small)
#          check  no API call: the tables and the frame search tool at each question's cutoff (needs the
#                 frame embeddings; the graph when it exists)
#          upload this video and arm's results folder to the Hugging Face dataset     (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    the agent's model, as in the paper; the memory is GPT-4.1 + SigLIP 2 for both
#          gpt41        GPT-4.1 agent
#          gemini25pro  Gemini 2.5 Pro agent (the paper's best)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
#
# Paid stages run only with EGAGENT_PAID_OK=1 and the key(s) in the environment (OPENAI_API_KEY for the
# memory and the gpt41 agent, GOOGLE_API_KEY for the gemini25pro agent): never in a file, an argument or a log.
# Without EGAGENT_PAID_OK=1 they print the calls they would make and stop (exit 3). epic/estimate_cost.py
# gives the cost.
#
# Environment (see README.md):
#   PY               python of the client environment (env/setup_env.sh)
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   EGAGENT_WORK     default baselines/egagent/work: frames (~0.16 MB each), per-question rows, tables, logs
#   EGAGENT_OUT      default baselines/egagent/results: per <arm>/<video> the memory, answers_<bank>.json.gz,
#                    blind_<bank>.json and runs.jsonl (`upload` puts them in the dataset
#                    morph-graphs/epic-morph-results at egagent/<arm>/<video>/)
#   EGAGENT_CHUNK_S  seconds of captions per graph extraction, default 300 (README: Streaming)
#   EGAGENT_TAG      a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   EGAGENT_FETCH    0 to never take the memory from the dataset (build normally reuses it when it is there)
#   EGAGENT_REPO_COMMIT, EGAGENT_UPSTREAM_COMMIT, EGAGENT_EPIC_COMMIT  commits for runs.jsonl when a checkout
#                    has no .git (a synced copy); read with git otherwise
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_egagent.sh <prep|build|qa|blind|check|upload|all> <video> <gpt41|gemini25pro> [bank]}
V=${2:?video id}; ARM=${3:?arm gpt41 or gemini25pro}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/egagent   # R: upstream
PY=${PY:?set PY to the python of the EGAgent client environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/egagent/langgraph_agent.py" ] || { echo "upstream missing: $R (README: Layout)"; exit 1; }
case $ARM in
    gpt41) BACKBONE=gpt-4.1; KEYS="OPENAI_API_KEY" ;;
    gemini25pro) BACKBONE=gemini-2.5-pro; KEYS="GOOGLE_API_KEY" ;;
    *) echo "arm must be gpt41 or gemini25pro"; exit 1 ;;
esac
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${EGAGENT_WORK:-$M/work}; OUTROOT=${EGAGENT_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; FRAMES=$WORK/frames/$V; TAG=${EGAGENT_TAG:-}; SUF=${TAG:+_$TAG}
MEMORY=("{video}_frames.npy" "{video}_captions.json" "{video}_graph.json")   # as epic/eg_paths.memory_files
mkdir -p "$OUT" "$WORK/logs"
export TOKENIZERS_PARALLELISM=false EPIC BACKBONE
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
have_memory() { local m; for m in "${MEMORY[@]}"; do [ -s "$OUT/${m//\{video\}/$V}" ] || return 1; done; }
paid() {   # $1: what would be called; the rest: the keys it needs
    local what=$1 k; shift
    if [ "${EGAGENT_PAID_OK:-}" != 1 ]; then
        echo "STOP before a paid API step: $what. Set EGAGENT_PAID_OK=1 (and $*) to run it."; exit 3
    fi
    for k in "$@"; do [ -n "${!k:-}" ] || { echo "$k is not set"; exit 1; }; done
}
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
commit = os.environ.get("EGAGENT_REPO_COMMIT") or git("rev-parse", "HEAD")
up = os.environ.get("EGAGENT_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/egagent/egagent")
bank = os.environ.get("EGAGENT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit,
                  "egagent_commit": up, "epic_morph_commit": bank, "agent_model": os.environ.get("BACKBONE"),
                  "memory_models": ["gpt-4.1", "google/siglip2-giant-opt-patch16-384"],
                  "chunk_s": int(os.environ.get("EGAGENT_CHUNK_S", 300)), "torch": torch.__version__,
                  "transformers": transformers.__version__, "gpus": gpus, "host": platform.node(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
cd "$REPO"

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4"; exit 1; }
    "$PY" "$M/epic/prepare_epic.py" --video "$VIDEOS/$P/$V.mp4" --frames_dir "$FRAMES" --threads "${THREADS:-8}"
    record prep $((SECONDS - t))
}
need_frames() { [ -s "$FRAMES/_done.json" ] || prep; }
embed() {   # the free, local part of the memory
    [ -s "$OUT/${V}_frames.npy" ] && return 0
    need_frames
    "$PY" "$M/epic/embed_frames.py" --frames_dir "$FRAMES" --out "$OUT/${V}_frames.npy"
}
build() {
    log build; local t=$SECONDS rc other f
    have_memory && { echo "memory complete in $OUT"; return 0; }
    # the memory depends only on the video (the same models for both arms): take a finished one from the other
    # arm's folder, then from the dataset (unless paid captions were started here), before building
    for other in "$OUTROOT"/*/"$V"; do
        [ "$other" != "$OUT" ] && [ -s "$other/${V}_frames.npy" ] && [ -s "$other/${V}_captions.json" ] \
            && [ -s "$other/${V}_graph.json" ] || continue
        for f in "${MEMORY[@]}"; do cp "$other/${f//\{video\}/$V}" "$OUT/"; done
        echo "memory copied from $other"; record "build (copied)" $((SECONDS - t)); return 0
    done
    if [ ! -s "$OUT/${V}_captions.json" ] && [ ! -e "$OUT/${V}_captions.json.partial" ] && [ "${EGAGENT_FETCH:-1}" != 0 ]; then
        rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline egagent --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -f "$OUT"/answers_*.json.gz; rm -rf "$WORK/rows/$ARM/$V"; record "build (fetched)" $((SECONDS - t)); return 0 ;;
            3) echo "no memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; EGAGENT_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    embed; need_frames
    if [ "${EGAGENT_PAID_OK:-}" != 1 ]; then   # say what the paid part would be before stopping
        "$PY" "$M/epic/caption_epic.py" --frames_dir "$FRAMES" --out "$OUT/${V}_captions.json" --dry
    fi
    paid "GPT-4.1 captions (one call per 30 s, 30 full-resolution frames each) and graph extraction (2 calls per ${EGAGENT_CHUNK_S:-300} s)" OPENAI_API_KEY
    [ -s "$OUT/${V}_captions.json" ] || "$PY" "$M/epic/caption_epic.py" --frames_dir "$FRAMES" --out "$OUT/${V}_captions.json"
    [ -s "$OUT/${V}_graph.json" ] || "$PY" "$M/epic/build_graph.py" --captions "$OUT/${V}_captions.json" --out "$OUT/${V}_graph.json" --chunk_s "${EGAGENT_CHUNK_S:-300}"
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
run_qa() { "$PY" -u "$M/epic/run_qa.py" --video "$V" --bank_file "$BF" --frames_dir "$FRAMES" --backbone "$BACKBONE" \
               --db_root "$WORK/dbs/${ARM}_${V}_$$" "$@"; }
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS k rc=0 ans=$OUT/answers_$BANK$SUF.json.gz
    [ -s "$ans" ] && { echo "done: $ans"; return 0; }
    have_memory || build
    need_frames; bank_file
    paid "the EGAgent workflow with $BACKBONE for every question of $BANK (about 12 calls per question, up to 50 frames per frame search)" $KEYS
    for k in 1 2 3; do   # a question that failed is retried by the next pass
        rc=0; run_qa --memory "$OUT" --rows "$WORK/rows/$ARM/$V/$BANK$SUF.jsonl" --out "$ans" || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "qa did not finish (exit $rc); rerun the same command to retry"; exit 1; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; local t=$SECONDS
    [ -s "$OUT/blind_$BANK.json" ] && { echo "done: $OUT/blind_$BANK.json"; return 0; }
    need_frames; bank_file
    paid "EGAgent's answer prompt with $BACKBONE, one short call per question of $BANK" $KEYS
    run_qa --blind --rows "$WORK/rows/$ARM/$V/blind_$BANK.jsonl" --out "$OUT/blind_$BANK.json"
    record "blind $BANK" $((SECONDS - t))
}
check() {   # no API call
    log "check $BANK"
    bank_file; embed
    BACKBONE=gpt-4.1 run_qa --memory "$OUT" --retrieval_check ${LIMIT:+--limit "$LIMIT"}
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline egagent --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; check) check ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, check, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
