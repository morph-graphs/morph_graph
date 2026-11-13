#!/usr/bin/env bash
# VST (Video Streaming Thinking) on one HD-EPIC video, without SLURM. See README.md.
#
#   bash baselines/vst/run_vst.sh <stage> <video> <arm> [bank]
#
#   stage  prep   nothing to make: VST reads the source video itself (checks that it is there)        CPU
#          build  writes the settings file. VST has no question-independent memory: its streaming thoughts
#                 depend on where the video is cut, so they are made in qa, per cutoff                 CPU
#          qa     per question cutoff (t_ask): streaming thoughts over the video up to it (cached), then
#                 VST answers every question of <bank> asked at that time                             GPU
#          blind  the same answer prompt with no video and no thoughts: the no-video floor            GPU
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    vst7b  Catalan258/VST-7B (Qwen2.5-VL-7B weights), VST's OVO-Bench settings     one GPU with >= 24 GB, 64 GB RAM
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing)
#
# Environment (see README.md):
#   PY               python of the VST environment (env/setup_env.sh vst7b <dir>)
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   VST_MODELS       a folder with a local copy of the model (VST-7B); default the Hugging Face id (cache)
#   VST_WORK         default baselines/vst/work: logs
#   VST_OUT          default baselines/vst/results: per <arm>/<video> the settings file, thoughts/ (per cutoff),
#                    answers_<bank>.json.gz, blind_<bank>.json.gz and runs.jsonl (`upload` puts them in the
#                    dataset morph-graphs/epic-morph-results at vst/<arm>/<video>/)
#   VST_TAG          a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   VST_THINK_TIMES, VST_VID_TOKENS, VST_MAX_FRAMES   other inference settings than the arm's (README.md); a
#                    results folder holds one setting, so give such a run its own VST_OUT
#   VST_LIMIT        smoke test: only the first N cutoffs of the bank (answers_<bank>_limit<N>.json.gz)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_vst.sh <prep|build|qa|blind|upload|all> <video> <vst7b> [bank]}
V=${2:?video id}; ARM=${3:?arm vst7b}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/VST   # R: the submodule
PY=${PY:?set PY to the python of the VST environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/eval/lmms-eval/lmms_eval/models/simple/qwen2_5_vl_stream_think.py" ] || { echo "submodule missing: git submodule update --init baselines/vst/VST"; exit 1; }
case $ARM in
    vst7b) NAME=Catalan258/VST-7B ;;
    *) echo "arm must be vst7b"; exit 1 ;;
esac
MODEL=$NAME; [ -z "${VST_MODELS:-}" ] || MODEL=$VST_MODELS/${NAME#*/}
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; VIDEO=$VIDEOS/${V%%-*}/$V.mp4
WORK=${VST_WORK:-$M/work}; OUT=${VST_OUT:-$M/results}/$ARM/$V
TAG=${VST_TAG:-}; SUF=${TAG:+_$TAG}${VST_LIMIT:+_limit$VST_LIMIT}
mkdir -p "$OUT" "$WORK/logs"
run_qa() {   # the adapter with this arm's model and settings
    "$PY" "$M/epic/run_qa.py" --video "$V" --out "$OUT" --model "$MODEL" --model_name "$NAME" \
        --think_times "${VST_THINK_TIMES:-2-3-5-5}" --vid_tokens "${VST_VID_TOKENS:-8192}" \
        --max_frames "${VST_MAX_FRAMES:-384}" "$@"
}
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'PYEOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("VST_REPO_COMMIT") or git("rev-parse", "HEAD")
sub = os.environ.get("VST_SUBMODULE_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/vst/VST")
bank = os.environ.get("VST_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/vst")) if not os.environ.get("VST_REPO_COMMIT") else None
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "vst_commit": sub, "epic_morph_commit": bank, "model": os.environ.get("MODEL"),
                  "torch": torch.__version__, "transformers": transformers.__version__, "gpus": gpus,
                  "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
PYEOF
}
export EPIC MODEL
cd "$REPO"

prep() {
    log prep
    [ -s "$VIDEO" ] || { echo "missing $VIDEO (epic-morph v3/DATA.md)"; exit 1; }
}
build() {
    log build
    run_qa --settings_only
}
bank_file() {   # this bank's questions for the video (banks are generated in epic-morph, not committed)
    local bank=${BANK:?qa and blind need a bank}; BF=$EPIC/outputs/v3/banks/$bank/${V}_questions.json
    if [ ! -s "$BF" ]; then
        mkdir -p "$EPIC/outputs/v3/banks"
        ( command -v flock >/dev/null && flock 9
          [ -s "$BF" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2
        ) 9>"$EPIC/outputs/v3/banks/.$bank.lock" || { echo "cannot generate bank $bank in $EPIC"; exit 1; }
    fi
    [ -s "$BF" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
}
answer() {   # answer <stage name> <answers file> [--blind]: up to three passes, each retrying what failed
    local name=$1 ans=$2 t=$SECONDS k rc=0; shift 2
    for k in 1 2 3; do
        rc=0; run_qa --video_file "$VIDEO" --bank_file "$BF" --answers "$ans" ${VST_LIMIT:+--limit "$VST_LIMIT"} "$@" || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "$name did not finish (exit $rc); rerun the same command to retry"; exit 1; }
    record "$name" $((SECONDS - t))
}
qa() {
    log "qa $BANK$SUF"
    prep; bank_file
    answer "qa $BANK$SUF" "$OUT/answers_$BANK$SUF.json.gz"
}
blind() {
    log "blind $BANK$SUF"
    bank_file
    answer "blind $BANK$SUF" "$OUT/blind_$BANK$SUF.json.gz" --blind
}
upload() {   # the settings file stands in for the memory: answers mean nothing without the settings that made them
    log upload
    [ -s "$OUT/${V}_settings.json" ] || { echo "no settings file in $OUT: run build or qa first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline vst --memory '{video}_settings.json' --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
