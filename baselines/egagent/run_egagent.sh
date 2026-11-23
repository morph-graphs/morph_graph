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
#          upload this video and arm's results folder to the Hugging Face dataset     (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#          check  (no API call) the tables and the frame search tool at each question's cutoff; needs the
#                 frame embeddings, uses the graph when it exists
#   arm    the agent's model, as in the paper; the memory is GPT-4.1 + SigLIP 2 for both
#          gpt41        GPT-4.1 agent           needs OPENAI_API_KEY
#          gemini25pro  Gemini 2.5 Pro agent    needs GEMINI_API_KEY (and OPENAI_API_KEY to build the memory)
#          gemini36flash  Gemini 3.6 Flash agent, not the paper's: it stands in for gemini25pro, which Google
#                       no longer serves to our key (README: Read first). Same keys as gemini25pro
#          OPENROUTER_API_KEY without OPENAI_API_KEY sends the GPT-4.1 calls through OpenRouter instead, and without
#          GEMINI_API_KEY the Gemini calls (README "Through OpenRouter"); with GEMINI_API_KEY Gemini is called through
#          Google's API (GOOGLE_GEMINI_BASE_URL: another endpoint that speaks it)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2
#
# A paid stage runs when its key is in the environment and stops (exit 3, nothing spent) when it is not. Keys
# are read from the environment only: never put one in a file, an argument or a log; a paid script is started
# without the other provider's key. Both models are being
# retired (README: Read first): the Gemini arm refuses to run from 2026-12-01 and GPT-4.1 calls from
# 2027-01-14, and a reply from any other model id stops the run (exit 4). epic/estimate_cost.py gives the cost.
#
# Environment (see README.md):
#   PY               python of the client environment (env/setup_env.sh)
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   EGAGENT_WORK     default baselines/egagent/work: frames (~0.16 MB each), per-question rows, tables, logs
#   EGAGENT_OUT      default baselines/egagent/results: per <arm>/<video> the memory, answers_<bank>.json.gz,
#                    blind_<bank>.json, api_usage.jsonl and runs.jsonl (`upload` puts them in the dataset
#                    morph-graphs/epic-morph-results at egagent/<arm>/<video>/)
#   EGAGENT_FRAME_PX frame width sent to the APIs and the embedder; default 0 = the video's own (1408 px), as
#                    published. Another value is another memory: give it its own EGAGENT_WORK and EGAGENT_OUT
#   EGAGENT_CHUNK_S  seconds of captions per graph extraction, default 300 (README: Read first)
#   EGAGENT_MEM_WORKERS  build: the memory's calls made at once (captions default 4, graph chunks default 1; q27: 8)
#   EGAGENT_TAG      a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   EGAGENT_FETCH    0 to never take the memory from the dataset (build normally reuses it when it is there)
#   EGAGENT_QA_WORKERS  qa: questions answered at once (threads), default 1 = one at a time (README: Parallel answering)
#   EGAGENT_REPO_COMMIT, EGAGENT_UPSTREAM_COMMIT, EGAGENT_EPIC_COMMIT  commits for runs.jsonl when a checkout
#                    has no .git (a synced copy); read with git otherwise
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/single_process_env.sh"   # never a torch.distributed job: drop torchrun's variables
STAGE=${1:?usage: run_egagent.sh <prep|build|qa|blind|upload|all|check> <video> <gpt41|gemini25pro|gemini36flash|g5m-g51|q27> [bank]}
V=${2:?video id}; ARM=${3:?arm gpt41, gemini25pro, gemini36flash, g5m-g51 or q27}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/egagent   # R: the submodule
PY=${PY:?set PY to the python of the EGAgent client environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
GPT_END=2027-01-14; GEMINI_END=2026-12-01   # the providers' shutdown dates (README: Read first)
case $ARM in
    gpt41) BACKBONE=gpt-4.1; KEY=OPENAI_API_KEY; END=$GPT_END ;;
    gemini25pro) BACKBONE=gemini-2.5-pro; KEY=GEMINI_API_KEY; END=$GEMINI_END ;;
    gemini36flash) BACKBONE=gemini-3.6-flash; KEY=GEMINI_API_KEY; END=9999-12-31 ;;   # no shutdown date announced
    # not the paper's, memory included: each has a memory of its own, built by MEMLLM
    g5m-g51) BACKBONE=gpt-5.1; KEY=OPENAI_API_KEY; END=9999-12-31; MEMLLM=gpt-5-mini; MEM_END=2026-12-11; MEMARMS=(g5m-g51) ;;
    # q27 sends 448 px frames (as MERIT's Qwen arms; at the video's own 1408 px a 30-frame caption call is 57k
    # tokens: 33 s a call and 74 GB on an H200) and makes its memory's calls 8 at a time (the local server batches them)
    q27) BACKBONE=Qwen/Qwen3.8-27B; KEY=LOCAL; END=9999-12-31; MEMLLM=$BACKBONE; MEM_END=$END; MEMARMS=(q27)
         EGAGENT_FRAME_PX=${EGAGENT_FRAME_PX:-448}; EGAGENT_MEM_WORKERS=${EGAGENT_MEM_WORKERS:-8} ;;
    *) echo "arm must be gpt41, gemini25pro, gemini36flash, g5m-g51 or q27"; exit 1 ;;
esac
# the memory's model, its shutdown date, and the arms whose folders hold that same memory (the paper's arms: GPT-4.1)
MEMLLM=${MEMLLM:-gpt-4.1}; MEM_END=${MEM_END:-$GPT_END}; [ -n "${MEMARMS+x}" ] || MEMARMS=(gpt41 gemini25pro gemini36flash)
MEM_KEY=OPENAI_API_KEY; [ "$KEY" != LOCAL ] || MEM_KEY=LOCAL
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${EGAGENT_WORK:-$M/work}; OUTROOT=${EGAGENT_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; TAG=${EGAGENT_TAG:-}; SUF=${TAG:+_$TAG}
PX=${EGAGENT_FRAME_PX:-0}; CHUNK=${EGAGENT_CHUNK_S:-300}; MEMW=${EGAGENT_MEM_WORKERS:-}
FRAMES=$WORK/frames/$V; [ "$PX" = 0 ] || FRAMES=$WORK/frames_$PX/$V   # resized frames in a folder of their own
MEMORY=("{video}_frames.npy" "{video}_captions.json" "{video}_graph.json")   # as epic/eg_paths.memory_files
mkdir -p "$OUT" "$WORK/logs"
[ -z "${OPENAI_API_KEY:-}" ] || unset OPENROUTER_API_KEY   # one gateway per run: OpenAI's key wins
export TOKENIZERS_PARALLELISM=false EPIC BACKBONE PX CHUNK EGAGENT_USAGE=$OUT/api_usage.jsonl EGAGENT_MEMORY_LLM=$MEMLLM
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
have_memory() { local m; for m in "${MEMORY[@]}"; do [ -s "$OUT/${m//\{video\}/$V}" ] || return 1; done; }

# upstream at its pinned commit + our patch (epic_morph.patch), applied once to the submodule's files
[ -f "$R/egagent/langgraph_agent.py" ] || { echo "upstream missing: git submodule update --init baselines/egagent/egagent"; exit 1; }
patched() { grep -q "def build_app" "$R/egagent/run_egagent_on_videomme.py" && grep -q EGAGENT_BACKBONE "$R/egagent/langgraph_agent.py" \
            && grep -q "startswith('gemini-')" "$R/egagent/langgraph_agent.py" \
            && grep -q EGAGENT_DB_ROOT "$R/paths.py" && grep -q "def get_api_key" "$R/utils.py" && grep -q EGAGENT_GPT_PREFIX "$R/utils.py" \
            && grep -q GEMINI_CLIENT_OPTIONS "$R/utils.py" && grep -q GEMINI_OPENROUTER "$R/utils.py" && grep -q "def is_chat_llm" "$R/utils.py"; }
if ! patched; then
    ( if command -v flock >/dev/null; then flock 9; fi
      if ! patched && grep -q OPENROUTER "$R/utils.py"; then   # an earlier epic_morph.patch: upstream's files back first
          git -C "$R" checkout -- . || { echo "an earlier epic_morph.patch is applied to $R: run git -C $R checkout -- ."; exit 1; }
          echo "restored upstream in $R (an earlier epic_morph.patch was applied)"
      fi
      # from outside any repository, so it behaves the same in a clone, a container and a synced copy
      patched || (cd / && git apply --unsafe-paths --whitespace=nowarn --directory="$R" "$M/epic_morph.patch")
    ) 9>"$WORK/.patch.lock" || true
    patched || { echo "epic_morph.patch does not apply to $R (is the submodule at its pinned commit, unmodified?)"; exit 1; }
    echo "applied epic_morph.patch to $R"
fi
# then epic_morph_trace.patch (logging only: each question's tool steps, run_qa.py's `trace`), on its own so a
# copy patched before it existed takes just this one
traced() { grep -q "^TRACE = \[\]" "$R/egagent/langgraph_agent.py"; }
if ! traced; then
    ( if command -v flock >/dev/null; then flock 9; fi
      traced || (cd / && git apply --unsafe-paths --whitespace=nowarn --directory="$R" "$M/epic_morph_trace.patch")
    ) 9>"$WORK/.patch.lock" || true
    traced || { echo "epic_morph_trace.patch does not apply to $R"; exit 1; }
    echo "applied epic_morph_trace.patch to $R"
fi

paid() {   # $1: the model, $2: its key's name (LOCAL: served here, no key), $3: its shutdown date, $4: what would be called
    [ "$2" != LOCAL ] || { [ -n "${EGAGENT_LOCAL_URL:-}" ] || { echo "STOP: $1 is served locally: set EGAGENT_LOCAL_URL to its OpenAI-compatible endpoint (docker/run.sh starts one)"; exit 3; }; return 0; }
    if [ "$(date -u +%F)" \> "$3" ] || [ "$(date -u +%F)" = "$3" ]; then
        echo "STOP: $1 is shut down from $3 (README: Read first); this stage cannot run as published any more."; exit 3
    fi
    if [ "$2" = OPENAI_API_KEY ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
        echo "$1 through OpenRouter (openai/$1, provider OpenAI only)"; return 0
    fi
    if [ "$2" = GEMINI_API_KEY ] && [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
        echo "$1 through OpenRouter (google/$1, provider Google AI Studio only)"; return 0
    fi
    [ -n "${!2:-}" ] || { echo "STOP before a paid API step, nothing spent: $4. It runs when $2 is set in the environment."; exit 3; }
}
record() {   # one line per finished stage, for the hand-back; API calls of the stage summed from api_usage.jsonl
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
commit = os.environ.get("EGAGENT_REPO_COMMIT") or git("rev-parse", "HEAD")
up = os.environ.get("EGAGENT_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/egagent/egagent")
bank = os.environ.get("EGAGENT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
api = {}   # model id the API returned -> calls and tokens, over every run of this stage
if os.path.exists(os.environ["EGAGENT_USAGE"]):
    for u in map(json.loads, open(os.environ["EGAGENT_USAGE"])):
        if u["stage"] == sys.argv[1]:
            for m, c in u["models"].items():   # counts, tokens and cost summed; endpoint and provider kept
                a = api.setdefault(m, {})
                for k, v in c.items():
                    a[k] = (v if isinstance(v, str) else sorted(set(a.get(k, [])) | set(v)) if isinstance(v, list)
                            else round(a.get(k, 0) + v, 6))
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit,
                  "egagent_commit": up, "epic_morph_commit": bank, "agent_model": os.environ.get("BACKBONE"),
                  "memory_models": [os.environ["EGAGENT_MEMORY_LLM"], "google/siglip2-giant-opt-patch16-384"], "api": api,
                  "frame_px": int(os.environ["PX"]), "chunk_s": int(os.environ["CHUNK"]), "torch": torch.__version__,
                  "transformers": transformers.__version__, "gpus": gpus, "host": platform.node(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
cd "$REPO"

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4"; exit 1; }
    "$PY" "$M/epic/prepare_epic.py" --video "$VIDEOS/$P/$V.mp4" --frames_dir "$FRAMES" --threads "${THREADS:-8}" --px "$PX"
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
    # the memory depends only on the video and the model that builds it (MEMARMS: the arms that share it): take a
    # finished one from another such arm's folder, then from the dataset (unless paid captions were started here)
    for other in "${MEMARMS[@]}"; do
        other=$OUTROOT/$other/$V
        [ "$other" != "$OUT" ] && [ -s "$other/${V}_frames.npy" ] && [ -s "$other/${V}_captions.json" ] \
            && [ -s "$other/${V}_graph.json" ] || continue
        for f in "${MEMORY[@]}"; do cp "$other/${f//\{video\}/$V}" "$OUT/"; done
        echo "memory copied from $other"; record "build (copied)" $((SECONDS - t)); return 0
    done
    if [ ! -s "$OUT/${V}_captions.json" ] && [ ! -e "$OUT/${V}_captions.json.partial" ] && [ "${EGAGENT_FETCH:-1}" != 0 ]; then
        for other in "${MEMARMS[@]}"; do   # any of these arms' folders in the dataset holds the same memory
            rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline egagent --memory "${MEMORY[@]}" \
                --arm "$other" --video "$V" --dir "$OUT" || rc=$?
            case $rc in
                0) rm -f "$OUT"/answers_*.json.gz; rm -rf "$WORK/rows/$ARM/$V"; record "build (fetched)" $((SECONDS - t)); return 0 ;;
                3) ;;
                *) echo "could not reach the dataset; EGAGENT_FETCH=0 builds the memory here instead"; exit 1 ;;
            esac
        done
        echo "no memory in the dataset yet: building it here"
    fi
    embed; need_frames
    [ -n "${OPENAI_API_KEY:-}${OPENROUTER_API_KEY:-}" ] || [ "$MEM_KEY" = LOCAL ] || "$PY" "$M/epic/caption_epic.py" --frames_dir "$FRAMES" --out "$OUT/${V}_captions.json" --dry
    paid "$MEMLLM" "$MEM_KEY" "$MEM_END" "$MEMLLM captions (one call per 30 s, 30 frames each) and graph extraction (2 calls per $CHUNK s)"
    export EGAGENT_USAGE_STAGE=build
    local gpt_only=(env -u GEMINI_API_KEY -u GOOGLE_API_KEY)   # the memory's calls are its model's alone
    [ "$MEM_KEY" != LOCAL ] || gpt_only=("${agent_only[@]}")
    [ -s "$OUT/${V}_captions.json" ] || "${gpt_only[@]}" "$PY" "$M/epic/caption_epic.py" --frames_dir "$FRAMES" --out "$OUT/${V}_captions.json" ${MEMW:+--workers "$MEMW"}
    [ -s "$OUT/${V}_graph.json" ] || "${gpt_only[@]}" "$PY" "$M/epic/build_graph.py" --captions "$OUT/${V}_captions.json" --out "$OUT/${V}_graph.json" --chunk_s "$CHUNK" ${MEMW:+--workers "$MEMW"}
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
agent_only=(env -u OPENAI_API_KEY -u OPENROUTER_API_KEY); [ "$KEY" != OPENAI_API_KEY ] || agent_only=(env -u GEMINI_API_KEY -u GOOGLE_API_KEY)
[ "$KEY" != GEMINI_API_KEY ] || [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] || agent_only=(env -u OPENAI_API_KEY)   # Gemini through OpenRouter
[ "$KEY" != LOCAL ] || agent_only=(env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY)   # no API at all
run_qa() { "${agent_only[@]}" "$PY" -u "$M/epic/run_qa.py" --video "$V" --bank_file "$BF" --frames_dir "$FRAMES" --backbone "$BACKBONE" \
               --db_root "$WORK/dbs/${ARM}_${V}_$$" "$@"; }
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS k rc=0 ans=$OUT/answers_$BANK$SUF.json.gz
    # answers from before rows carried a trace are not done: their questions are asked again
    [ -s "$ans" ] && "$PY" "$M/../common/earlier.py" traced "$ans" && { echo "done: $ans"; return 0; }
    have_memory || build
    need_frames; bank_file
    paid "$BACKBONE" "$KEY" "$END" "the EGAgent workflow with $BACKBONE for every question of $BANK (about 12 calls per question, up to 50 frames per frame search)"
    export EGAGENT_USAGE_STAGE="qa $BANK$SUF"
    for k in 1 2 3; do   # a question that failed is retried by the next pass
        rc=0; run_qa --memory "$OUT" --rows "$WORK/rows/$ARM/$V/$BANK$SUF.jsonl" --out "$ans" ${LIMIT:+--limit "$LIMIT"} || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "qa did not finish (exit $rc); rerun the same command to retry"; exit "$rc"; }
    [ -z "${LIMIT:-}" ] || { echo "LIMIT=$LIMIT: the first questions only, no answers file; the same command without LIMIT finishes"; return 0; }
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; local t=$SECONDS
    [ -s "$OUT/blind_$BANK.json" ] && { echo "done: $OUT/blind_$BANK.json"; return 0; }
    need_frames; bank_file
    paid "$BACKBONE" "$KEY" "$END" "EGAgent's answer prompt with $BACKBONE, one short call per question of $BANK"
    export EGAGENT_USAGE_STAGE="blind $BANK"
    run_qa --blind --rows "$WORK/rows/$ARM/$V/blind_$BANK.jsonl" --out "$OUT/blind_$BANK.json"
    record "blind $BANK" $((SECONDS - t))
}
check() {   # no API call
    log "check $BANK"
    bank_file; embed; need_frames
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
    *) echo "stage must be prep, build, qa, blind, upload, all or check"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
