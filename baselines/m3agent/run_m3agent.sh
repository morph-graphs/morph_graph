#!/usr/bin/env bash
# M3-Agent (ByteDance Seed, ICLR 2026) on one HD-EPIC video, without SLURM. See README.md here.
#
#   bash baselines/m3agent/run_m3agent.sh <stage> <video> <arm> [bank]
#
#   stage  prep   30 s clips with audio (720 px) + clips.json                               CPU, ffmpeg
#          build  the memory graph: per clip faces, voices + ASR, then the memorization       one GPU >= 40 GB
#                 model's episodic and semantic memories (no questions needed)
#          qa     the control model answers every question of <bank> for this video, each     2x 48 GB (TP=2)
#                 from the memory truncated at its t_ask (needs build)                        or one 80 GB GPU
#          blind  the control model with no memory: the no-memory floor (no build needed)    as qa
#          upload this video's results folder to the Hugging Face dataset (m3agent/<arm>/<video>/)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    m3     the published checkpoints (M3-Agent-Memorization, M3-Agent-Control)
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing)
#
# Environment (see README.md):
#   PY_MEM, PY_CTL  pythons of the two environments (env/setup_env.sh mem|ctl): memorization (prep, build) and
#                   control (qa, blind, upload; also the embedding model's helper process during build)
#   EPIC_MORPH_DIR  an epic-morph checkout (required for qa and blind): the banks, and by default the videos
#   HDEPIC_VIDEOS   default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   M3AGENT_WORK    default baselines/m3agent/work: clips, intermediate outputs (faces, voices), control runs
#   M3AGENT_OUT     default baselines/m3agent/results: per <arm>/<video> the memory ({video}_m3graph.pkl,
#                   {video}_memories.json), answers_<bank>.json.gz, blind_<bank>.json and runs.jsonl
#   M3AGENT_MODELS  default $M3AGENT_WORK/models: M3-Agent-Memorization, M3-Agent-Control, Qwen3-Embedding-4B,
#                   faster-whisper-large-v3-turbo, pretrained_eres2netv2.ckpt, insightface/ (each overridable:
#                   M3AGENT_MEM_CKPT, M3AGENT_CONTROL_CKPT, M3AGENT_EMBED_MODEL, M3AGENT_ASR_MODEL,
#                   M3AGENT_ERES2NET, M3AGENT_INSIGHTFACE_ROOT)
#   M3AGENT_BACKEND local (default: local ASR and embeddings) or api (upstream's Gemini / OpenAI calls; untested)
#   M3AGENT_TAG     a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   M3AGENT_FETCH   0 to never take the memory from the dataset (build normally reuses it when it is there)
#   M3AGENT_GPU_UTIL, M3AGENT_MAX_LEN  vLLM's gpu_memory_utilization (0.8) and max_model_len (16384)
# Every stage skips finished work, so rerunning the same command resumes (build resumes after its last clip).
set -euo pipefail
STAGE=${1:?usage: run_m3agent.sh <prep|build|qa|blind|upload|all> <video> <arm> [bank]}
V=${2:?video id}; ARM=${3:?arm: m3}; BANK=${4:-}
[ "$ARM" = m3 ] || { echo "arm must be m3"; exit 1; }
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/m3-agent   # R: the submodule
[ -f "$R/m3_agent/control.py" ] || { echo "m3-agent submodule missing: git submodule update --init baselines/m3agent/m3-agent"; exit 1; }
WORK=${M3AGENT_WORK:-$M/work}; OUTROOT=${M3AGENT_OUT:-$M/results}; MODELS=${M3AGENT_MODELS:-$WORK/models}
OUT=$OUTROOT/$ARM/$V; CLIPS=$WORK/clips/$V; MEM=$OUT/${V}_m3graph.pkl; MEMJSON=$OUT/${V}_memories.json
TAG=${M3AGENT_TAG:-}; SUF=${TAG:+_$TAG}
VIDEOS=${HDEPIC_VIDEOS:-${EPIC_MORPH_DIR:+$EPIC_MORPH_DIR/data/HD-EPIC/videos}}
export M3AGENT_MEM_CKPT=${M3AGENT_MEM_CKPT:-$MODELS/M3-Agent-Memorization}
export M3AGENT_CONTROL_CKPT=${M3AGENT_CONTROL_CKPT:-$MODELS/M3-Agent-Control}
export M3AGENT_EMBED_MODEL=${M3AGENT_EMBED_MODEL:-$MODELS/Qwen3-Embedding-4B}
export M3AGENT_ASR_MODEL=${M3AGENT_ASR_MODEL:-$MODELS/faster-whisper-large-v3-turbo}
export M3AGENT_ERES2NET=${M3AGENT_ERES2NET:-$MODELS/pretrained_eres2netv2.ckpt}
export M3AGENT_INSIGHTFACE_ROOT=${M3AGENT_INSIGHTFACE_ROOT:-$MODELS/insightface}
export M3AGENT_BACKEND=${M3AGENT_BACKEND:-local} MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$WORK/logs"
NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'), $NGPU GPU)"; }
need() { local v=$1; [ -n "${!v:-}" ] || { echo "set $v to the python of the ${2} environment (env/setup_env.sh)"; exit 1; }; }
record() {   # one line per finished stage: record <python> <stage> <seconds>
    "$1" - "$2" "$3" "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
repo = os.environ["REPO"]
commit = os.environ.get("M3AGENT_REPO_COMMIT") or git("rev-parse", "HEAD", cwd=repo)
sub = os.environ.get("M3AGENT_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd=os.path.join(repo, "baselines/m3agent/m3-agent"))
epic = os.environ.get("EPIC_MORPH_DIR")
bank = os.environ.get("M3AGENT_EPIC_COMMIT") or (git("rev-parse", "HEAD", cwd=epic) if epic else None)
dirty = bool(git("status", "--porcelain", "baselines/m3agent", cwd=repo)) if git("rev-parse", "HEAD", cwd=repo) else None   # None: no git here
try:
    from importlib.metadata import version; vv = version("vllm")   # not imported: vllm logs to stdout
except Exception:
    vv = None
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
open(sys.argv[3], "a").write(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "m3agent_commit": sub, "epic_morph_commit": bank,
                  "version": open(os.path.join(repo, "baselines/m3agent/VERSION")).read().strip(),
                  "backend": os.environ.get("M3AGENT_BACKEND"), "torch": torch.__version__,
                  "transformers": transformers.__version__, "vllm": vv,
                  "gpus": gpus, "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
EOF
}
export REPO
cd "$REPO"

prep() {
    log prep; need PY_MEM memorization; local t=$SECONDS P=${V%%-*}
    [ -s "$CLIPS/clips.json" ] && return 0
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4 (set HDEPIC_VIDEOS or EPIC_MORPH_DIR)"; exit 1; }
    PATH="$(dirname "$PY_MEM"):$PATH" "$PY_MEM" "$M/epic/m3_epic.py" clips --video "$VIDEOS/$P/$V.mp4" --clip_dir "$CLIPS" --threads "${THREADS:-8}"
    record "$PY_MEM" prep $((SECONDS - t))
}
build() {
    log build; local t=$SECONDS
    [ -s "$MEM" ] && [ -s "$MEMJSON" ] && { echo "memory already built: $MEM"; return 0; }
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local build is under way (a half-built local memory is resumed, never overwritten)
    if [ ! -s "$WORK/partial/$V.pkl" ] && [ "${M3AGENT_FETCH:-1}" != 0 ]; then
        need PY_CTL control; local rc=0
        "$PY_CTL" "$M/../common/hf_results.py" fetch-memory --baseline m3agent --memory '{video}_m3graph.pkl' '{video}_memories.json' \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -rf "$WORK/runs/$V"; rm -f "$OUT"/answers_*.json.gz; return 0 ;;   # answers from an earlier memory are stale
            3) echo "no memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; M3AGENT_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    need PY_MEM memorization; need PY_CTL control
    [ -s "$CLIPS/clips.json" ] || prep
    mkdir -p "$WORK/partial" "$WORK/intermediate/$V"
    local data=$WORK/partial/$V.jsonl
    "$PY_MEM" -c "import json,sys; print(json.dumps(dict(zip(['id','clip_path','mem_path','json_path','partial_path','intermediate_outputs'], sys.argv[1:]))))" \
        "$V" "$CLIPS" "$MEM" "$MEMJSON" "$WORK/partial/$V.pkl" "$WORK/intermediate/$V" > "$data"
    rm -f "$MEM" "$MEMJSON"   # one without the other is from an interrupted write
    # upstream reads configs/ from the working directory; the embedding model runs in the control environment
    (cd "$R" && PATH="$(dirname "$PY_MEM"):$PATH" M3AGENT_EMBED_PY=$PY_CTL "$PY_MEM" m3_agent/memorization_memory_graphs.py --data_file "$data")
    [ -s "$MEM" ] && [ -s "$MEMJSON" ] || { echo "memorization did not finish: $MEM"; exit 1; }
    record "$PY_MEM" build $((SECONDS - t))
}
qa_file() {   # this bank's questions for the video in M3-Agent's control format, with before_clip from t_ask
    local bank=${BANK:?qa and blind need a bank}; EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
    local src=$EPIC/outputs/v3/banks/$bank/${V}_questions.json; QA=$WORK/qa/$bank/$V.json
    if [ ! -s "$src" ]; then   # banks are generated, not committed; one generator at a time
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$src" ] || (cd "$EPIC" && "$PY_CTL" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$src" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
    [ -s "$CLIPS/clips.json" ] || prep
    mkdir -p "$(dirname "$QA")"
    "$PY_CTL" "$M/epic/m3_epic.py" questions --bank "$src" --video_id "$V" --clips "$CLIPS/clips.json" --mem_path "$MEM" --out "$QA"
}
control() {   # control <run jsonl> [--blind]: answer the questions not yet in the run file (resumes)
    local tp=${M3AGENT_TP:-$NGPU}
    [ "$tp" -ge 1 ] || { echo "no GPU visible"; exit 1; }
    (cd "$R" && "$PY_CTL" m3_agent/control.py --data_file "$QA" --output "$1" --model "$M3AGENT_CONTROL_CKPT" --tp "$tp" \
        --gpu_memory_utilization "${M3AGENT_GPU_UTIL:-0.8}" --max_model_len "${M3AGENT_MAX_LEN:-16384}" "${@:2}")
}
qa() {
    log "qa $BANK$SUF"; need PY_CTL control; local t=$SECONDS run=$WORK/runs/$V/${BANK}${SUF}.jsonl missing
    [ -s "$MEM" ] || build
    qa_file; mkdir -p "$(dirname "$run")"
    for k in 1 2; do   # a crashed pass leaves the questions answered so far; the second pass does the rest
        control "$run" 2>&1 | tee -a "$WORK/logs/${V}_${BANK}${SUF}.log" || echo "pass $k exited with an error"
        missing=$("$PY_CTL" "$M/epic/m3_epic.py" collect --questions "$QA" --clips "$CLIPS/clips.json" --runs "$run" --mem_path "$MEM" \
            --out "$OUT/answers_$BANK$SUF.json.gz" | tee /dev/stderr | tail -1)
        [ -z "$missing" ] && break
    done
    [ -z "$missing" ] || { echo "still unanswered: $missing (rerun the same command to retry)"; exit 1; }
    record "$PY_CTL" "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; need PY_CTL control; local t=$SECONDS run=$WORK/runs/$V/blind_${BANK}.jsonl
    qa_file; mkdir -p "$(dirname "$run")"
    control "$run" --blind 2>&1 | tee -a "$WORK/logs/${V}_blind_${BANK}.log"
    local missing; missing=$("$PY_CTL" "$M/epic/m3_epic.py" collect --questions "$QA" --clips "$CLIPS/clips.json" --runs "$run" \
        --out "$OUT/blind_$BANK.json" | tee /dev/stderr | tail -1)
    [ -z "$missing" ] || { echo "blind: still unanswered: $missing"; exit 1; }
    record "$PY_CTL" "blind $BANK" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload; need PY_CTL control
    [ -s "$MEM" ] && [ -s "$MEMJSON" ] || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY_CTL" "$M/../common/hf_results.py" upload --baseline m3agent --memory '{video}_m3graph.pkl' '{video}_memories.json' \
        --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
