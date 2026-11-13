#!/usr/bin/env bash
# Direct baseline on one HD-EPIC video, without SLURM: a plain video model with no memory. For each question
# the model sees frames sampled from the video before the question's ask time, and answers. See README.md here.
#
#   bash baselines/direct/run_direct.sh <stage> <video> <arm> [bank]
#
#   stage  prep   the 1 fps, 448 px frames of the video: MERIT's prep, shared with it      CPU, ffmpeg
#          build  nothing: there is no memory (kept so every baseline has the same stages)
#          qa     the model answers every question of <bank> for this video                GPU
#          upload this video and arm's results folder to the Hugging Face dataset          (not wired yet, see README)
#          all    prep, then qa when a bank is given (no upload)
#   arm    <model>-f<N>[-p<px>]: the model, the frame budget N, and the frame side when it is not 448
#          model  qwen3vl-8b   Qwen/Qwen3-VL-8B-Instruct   MERIT's m8 environment
#                 qwen38-27b   Qwen/Qwen3.8-27B            MERIT's m27 environment
#          N      at most N frames per question, evenly spaced over the video before t_ask, the last one
#                 just before t_ask; all of them (1 fps) when the video so far has fewer than N seconds
#          e.g. qwen3vl-8b-f32, qwen3vl-8b-f128, qwen38-27b-f768, qwen3vl-8b-f768-p224
#   bank   a named bank of the epic-morph checkout, e.g. batch10_agree2 (generated there with
#          v3/export_bank.py <bank> when missing)
#
# Environment: PY  python of the model's environment (baselines/merit/env/setup_env.sh m8 or m27)
#   EPIC_MORPH_DIR an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS  default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   DIRECT_WORK    default baselines/merit/work: MERIT's prep folder, so a video is cut once for both
#   DIRECT_OUT     default baselines/direct/results: per <arm>/<video>, answers_<bank>.json.gz and runs.jsonl
#   DIRECT_LIMIT   answer only this many questions, spread over the bank (smoke test)
#   DIRECT_REPO_COMMIT, DIRECT_MERIT_COMMIT, DIRECT_EPIC_COMMIT  commits for runs.jsonl when a checkout has no .git
#                  (a synced copy); read with git otherwise
# qa skips answered questions, so rerunning the same command resumes.
set -euo pipefail
STAGE=${1:?usage: run_direct.sh <prep|build|qa|upload|all> <video> <model>-f<N>[-p<px>] [bank]}
V=${2:?video id}; ARM=${3:?arm, e.g. qwen3vl-8b-f128}; BANK=${4:-}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); MERIT=$M/../merit
PY=${PY:?set PY to the python of the model\'s environment}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
[ -f "$MERIT/MERIT/eval/evaluate_egolife.py" ] || { echo "MERIT submodule missing (its model loader, prompt and scorer are used): git submodule update --init baselines/merit/MERIT"; exit 1; }
[[ $ARM =~ ^(qwen3vl-8b|qwen38-27b)-f([0-9]+)(-p([0-9]+))?$ ]] || { echo "arm must be <qwen3vl-8b|qwen38-27b>-f<N>[-p<px>], got $ARM"; exit 1; }
MODEL=${BASH_REMATCH[1]}; FRAMES=${BASH_REMATCH[2]}; PX=${BASH_REMATCH[4]:-448}
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}; WORK=${DIRECT_WORK:-$MERIT/work}; OUTROOT=${DIRECT_OUT:-$M/results}
OUT=$OUTROOT/$ARM/$V; SKEL=$WORK/prep/captions/$V/${V}_30sec_skeleton.json
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# as run_merit.sh: one GPU for the 8B; the 27B spreads over the GPUs it is given
if [ "$STAGE" != prep ] && [ "$STAGE" != build ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$( (nvidia-smi -L 2>/dev/null || true) | wc -l)" -gt 1 ]; then
    echo "several GPUs visible: set CUDA_VISIBLE_DEVICES to the GPU(s) for this job (e.g. CUDA_VISIBLE_DEVICES=0)"; exit 1
fi
if [[ ${CUDA_VISIBLE_DEVICES:-} == *,* ]]; then
    export MERIT_DEVICE_MAP=auto
    [ "$MODEL" = qwen38-27b ] && export MERIT_GPU0_FRAC=0.6 CUDA_LAUNCH_BLOCKING=1
fi
log() { echo "=== $V $ARM $1 ($(date '+%F %H:%M'))"; }
record() {   # one line per finished stage
    "$PY" - "$1" "$2" >> "$OUT/runs.jsonl" <<'EOF'
import json, os, platform, subprocess, sys, time
import torch, transformers
git = lambda *a, cwd=None: subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd).stdout.strip()
print(json.dumps({"stage": sys.argv[1], "seconds": int(float(sys.argv[2])),
                  "commit": os.environ.get("DIRECT_REPO_COMMIT") or git("rev-parse", "HEAD"),
                  "dirty": bool(git("status", "--porcelain", "baselines/direct", "baselines/merit")),
                  "merit_commit": os.environ.get("DIRECT_MERIT_COMMIT") or git("rev-parse", "HEAD", cwd="baselines/merit/MERIT"),
                  "epic_morph_commit": os.environ.get("DIRECT_EPIC_COMMIT") or git("rev-parse", "HEAD", cwd=os.environ["EPIC"]),
                  "model": os.environ["MODEL"], "frames": int(os.environ["FRAMES"]), "frame_px": int(os.environ["PX"]),
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  "host": platform.node(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
}
export MODEL FRAMES PX EPIC
cd "$REPO"

prep() {   # MERIT's prep with MERIT's lock and settings (run_merit.sh prep), so both use one set of frames
    log prep; local P=${V%%-*}
    [ -s "$SKEL" ] && return 0
    [ -s "$VIDEOS/$P/$V.mp4" ] || { echo "missing $VIDEOS/$P/$V.mp4 (v3/DATA.md)"; exit 1; }
    mkdir -p "$WORK/locks"
    (
        flock 9
        [ -s "$SKEL" ] && exit 0
        "$PY" "$MERIT/epic/prepare_epic.py" --video "$VIDEOS/$P/$V.mp4" --subject "$V" --clip_root "$WORK/clips_448" \
            --frame_root "$WORK/frames_448" --data_dir "$WORK/prep" --clip_size 448 --frame_size 448 --threads "${THREADS:-8}"
    ) 9>"$WORK/locks/prep_$V.lock"
}
qa() {
    local bank=${BANK:?qa needs a bank} t=$SECONDS; log "qa $bank"
    local bf=$EPIC/outputs/v3/banks/$bank/${V}_questions.json
    if [ ! -s "$bf" ]; then   # banks are generated, not committed; one generator at a time (as run_merit.sh)
        mkdir -p "$EPIC/outputs/v3/banks"
        ( flock 9; [ -s "$bf" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$bank") >&2 ) 9>"$EPIC/outputs/v3/banks/.$bank.lock"
    fi
    [ -s "$bf" ] || { echo "no $bank questions for $V in $EPIC"; exit 1; }
    prep
    "$PY" "$M/epic/direct_qa.py" --model "$MODEL" --frames "$FRAMES" --frame_px "$PX" --video "$V" --bank_file "$bf" \
        --frame_root "$WORK/frames_448" --skeleton "$SKEL" --output "$OUT/answers_$bank.json.gz" ${DIRECT_LIMIT:+--limit "$DIRECT_LIMIT"} \
        || { echo "some questions failed (out of GPU memory): rerun the same command to retry them"; exit 1; }
    record "qa $bank" $((SECONDS - t))
}
upload() {
    # baselines/common/hf_results.py uploads only a folder that holds a memory; this baseline has none.
    # It needs a no-memory mode there first (outside this folder), see README.md
    echo "upload is not wired for the direct baseline yet: results stay in $OUT"; exit 1
}

case $STAGE in
    prep) prep ;; build) echo "direct has no memory to build" ;; qa) qa ;; upload) upload ;;
    all) prep; [ -z "$BANK" ] || qa ;;
    *) echo "stage must be prep, build, qa, upload or all"; exit 1 ;;
esac
echo "=== $V $ARM $STAGE done"
