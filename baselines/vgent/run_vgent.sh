#!/usr/bin/env bash
# Vgent on one HD-EPIC video, without SLURM. See README.md.
#
#   bash baselines/vgent/run_vgent.sh <stage> <video> <arm> [bank]
#
#   stage  prep   the video's 1 fps frames at Vgent's size, as one memory-mapped file      CPU (decord)
#          build  Vgent's offline graph: one backbone call per 64 s chunk, cached            GPU
#          qa     Vgent answers every question of <bank> for this video, each from the
#                 frames and the graph up to its t_ask                                       GPU (needs build)
#          blind  the backbone on the question alone, same final prompt: the no-memory floor GPU (no build needed)
#          upload this video and arm's results folder to the Hugging Face dataset  (needs a write token)
#          all    prep + build, then qa + blind when a bank is given (no upload)
#   arm    Vgent's own backbone, by its name in upstream's MODEL_MAP. The paper's configurations that run on
#          upstream's Qwen-VL wrapper: qwenvl25_7b (Qwen2.5-VL-7B-Instruct, upstream's default; the one
#          tested here), qwenvl25_3b, qwenvl2_7b, qwenvl2_2b. One GPU with >= 40 GB.
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing)
#
# Environment (see README.md):
#   PY              python of the Vgent environment (env/setup_env.sh)
#   EPIC_MORPH_DIR  an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS   default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   VGENT_WORK      default baselines/vgent/work: the frames (~1.1 MB per second of video) and locks
#   VGENT_OUT       default baselines/vgent/results: per <arm>/<video> the memory (<video>_nodes.jsonl,
#                   <video>_graph.pkl), <video>_cutoff_nodes.jsonl, answers_<bank>.json.gz, blind_<bank>.json and
#                   runs.jsonl (`upload` puts them in the dataset morph-graphs/epic-morph-results at vgent/<arm>/<video>/)
#   VGENT_TAG       a variant name for the answers (answers_<bank>_<tag>.json.gz)
#   VGENT_MAX_QUESTIONS  answer only the first N unanswered questions (smoke test)
#   VGENT_FETCH     0 to never take the memory from the dataset (build normally reuses it when it is there)
# Every step skips finished work, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_vgent.sh <prep|build|qa|blind|upload|all> <video> <arm> [bank]}
V=${2:?video id}; ARM=${3:?arm, e.g. qwenvl25_7b}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); R=$M/Vgent   # R: the submodule (upstream, unpatched)
PY=${PY:?set PY to the python of the Vgent environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$R/utils/vgent.py" ] || { echo "submodule missing: git submodule update --init baselines/vgent/Vgent"; exit 1; }
case $ARM in qwenvl25_7b|qwenvl25_3b|qwenvl2_7b|qwenvl2_2b) ;; *) echo "arm must be qwenvl25_7b, qwenvl25_3b, qwenvl2_7b or qwenvl2_2b"; exit 1 ;; esac
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${VGENT_WORK:-$M/work}; OUTROOT=${VGENT_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; FRAMES=$WORK/frames
TAG=${VGENT_TAG:-}; SUF=${TAG:+_$TAG}
mkdir -p "$OUT" "$WORK/locks"
export VGENT_SRC=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read -ra MEMORY <<< "$("$PY" "$M/epic/build_graph.py" --memory_files)"
have_memory() { local f; for f in "${MEMORY[@]}"; do [ -s "$OUT/${f//\{video\}/$V}" ] || return 1; done; }
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage, for the hand-back
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'PYEOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
# a checkout without .git (a synced copy, a container) passes the commits read elsewhere
commit = os.environ.get("VGENT_REPO_COMMIT") or git("rev-parse", "HEAD")
sub = os.environ.get("VGENT_UPSTREAM_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/vgent/Vgent")
bank = os.environ.get("VGENT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"])
dirty = bool(git("status", "--porcelain", "baselines/vgent")) if not os.environ.get("VGENT_REPO_COMMIT") else None
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])), "commit": commit, "dirty": dirty,
                  "vgent_commit": sub, "epic_morph_commit": bank, "model": os.environ.get("ARM"),
                  "torch": torch.__version__, "transformers": transformers.__version__, "gpus": gpus,
                  "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
PYEOF
}
export EPIC ARM
cd "$REPO"

prep() {
    log prep; local t=$SECONDS P=${V%%-*}
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4"; exit 1; }
    # one prep per video at a time: two arms of the same video share the frames
    ( flock 9; "$PY" "$M/epic/prepare_frames.py" --video "$VIDEOS/$P/$V.mp4" --out "$FRAMES" --name "$V" --threads "${THREADS:-8}"
    ) 9>"$WORK/locks/prep_$V.lock"
    record prep $((SECONDS - t))
}
need_frames() { [ -s "$FRAMES/$V.json" ] && [ -s "$FRAMES/$V.npy" ] || prep; }
build() {
    log build; local t=$SECONDS rc
    # the memory depends only on the video and the model: take it from the dataset when it is there and no
    # local one is started (a half-built local memory is resumed, never overwritten)
    if [ ! -s "$OUT/${V}_nodes.jsonl" ] && [ "${VGENT_FETCH:-1}" != 0 ]; then
        rc=0; "$PY" "$M/../common/hf_results.py" fetch-memory --baseline vgent --memory "${MEMORY[@]}" \
            --arm "$ARM" --video "$V" --dir "$OUT" || rc=$?
        case $rc in
            0) rm -f "$OUT"/answers_*.json.gz "$OUT/${V}_cutoff_nodes.jsonl" ;;   # made from an earlier local memory
            3) echo "no Vgent memory in the dataset yet: building it here" ;;
            *) echo "could not reach the dataset; VGENT_FETCH=0 builds the memory here instead"; exit 1 ;;
        esac
    fi
    if ! have_memory; then
        need_frames
        "$PY" "$M/epic/build_graph.py" --video "$V" --arm "$ARM" --frames "$FRAMES" --out "$OUT"
    fi
    have_memory || { echo "memory for $V is incomplete in $OUT"; exit 1; }
    record build $((SECONDS - t))
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
answer() {   # answer <output file> [run_qa.py options]: up to three passes over what has no valid answer yet
    local out=$1 k rc=0; shift
    for k in 1 2 3; do
        rc=0; "$PY" "$M/epic/run_qa.py" --video "$V" --arm "$ARM" --bank_file "$BF" --frames "$FRAMES" \
            --answers "$out" ${VGENT_MAX_QUESTIONS:+--max_questions "$VGENT_MAX_QUESTIONS"} "$@" || rc=$?
        [ $rc = 2 ] || break
    done
    [ $rc = 0 ] || { echo "$STAGE did not finish (exit $rc); rerun the same command to retry"; exit 1; }
}
qa() {
    log "qa $BANK$SUF"; local t=$SECONDS
    have_memory || build
    need_frames; bank_file   # a memory fetched from the dataset comes without the frames
    answer "$OUT/answers_$BANK$SUF.json.gz" --memory "$OUT"
    record "qa $BANK$SUF" $((SECONDS - t))
}
blind() {
    log "blind $BANK"; local t=$SECONDS
    bank_file
    answer "$OUT/blind_$BANK.json" --blind
    record "blind $BANK" $((SECONDS - t))
}
upload() {   # only a finished memory: a half-built folder would overwrite a good one in the dataset
    log upload
    have_memory || { echo "no finished memory in $OUT: run build first"; exit 1; }
    "$PY" "$M/../common/hf_results.py" upload --baseline vgent --memory "${MEMORY[@]}" --arm "$ARM" --video "$V" --dir "$OUT"
}

case $STAGE in
    prep) prep ;; build) build ;; qa) qa ;; blind) blind ;; upload) upload ;;
    all) prep; build; [ -z "$BANK" ] || { qa; blind; } ;;
    *) echo "stage must be prep, build, qa, blind, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
