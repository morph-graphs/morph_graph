#!/usr/bin/env bash
# One video of a bank through one closed-source API model (Gemini, GPT), with epic_eval's steps and no Slurm.
# Every stage skips finished work, so the same command resumes after a failure.
#   bash baselines/api/run_api.sh <stage> <video> <config> [bank]
#   stage   prep    cut the video's [0, t_ask] clips (ffmpeg, CPU; shared by every config with the same media settings)
#           qa      cut the clips if missing, ask the model every unanswered question of the video, then write the answers file
#           upload  put the video's results folder in the Hugging Face dataset (api/<config>/<video>/)
#           score   epic_eval's score table over the whole run folder
#           all     prep, qa
#   config  a file name in baselines/api/configs/ without .json, e.g. gemini-3.6-flash-high, gpt-5-f1024;
#           it is also the arm name of the results
#   bank    default batch10_agree2 (epic-morph outputs/v3/banks/<bank>/; made there by v3/export_bank.py when missing)
# Environment:
#   EPIC_MORPH_DIR   an epic-morph checkout (required): the banks, and by default the videos
#   HDEPIC_VIDEOS    default $EPIC_MORPH_DIR/data/HD-EPIC/videos (<P>/<video>.mp4)
#   GEMINI_API_KEY / OPENAI_API_KEY   the key of the config's provider (qa only; a paid-tier key)
#   API_LIMIT        qa: ask at most this many questions (a smoke test)
#   API_WORK         run folders and clips (default baselines/api/work, not committed)
#   API_OUT          results root (default baselines/api/results, not committed): <config>/<video>/answers_<bank>.json.gz, runs.jsonl
#   API_MEDIA_DIR    reuse a folder of clips already encoded with the same media settings
#   PY               the Python to use (default python3; needs env/requirements.txt, and ffmpeg on PATH)
set -euo pipefail
STAGE=${1:?usage: run_api.sh <prep|qa|upload|score|all> <video> <config> [bank]}
V=${2:?video id}; CFG=${3:?config name}; BANK=${4:-batch10_agree2}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$M/../.." && pwd); PY=${PY:-python3}
EPIC=$(cd "${EPIC_MORPH_DIR:?set EPIC_MORPH_DIR to an epic-morph checkout}" && pwd)
VIDEOS=${HDEPIC_VIDEOS:-$EPIC/data/HD-EPIC/videos}
WORK=${API_WORK:-$M/work}; OUT=${API_OUT:-$M/results}
[ -f "$M/configs/$CFG.json" ] || { echo "no config $M/configs/$CFG.json; have: $(cd "$M/configs" && ls | sed 's/\.json$//' | tr '\n' ' ')"; exit 1; }
cd "$REPO"   # epic_eval is run as a module from the repo root
log() { echo "[$(date +%H:%M:%S)] $CFG $V: $*"; }

setup() {
    mkdir -p "$WORK/locks" "$EPIC/outputs/v3/banks"
    # one bank export at a time (tasks of other videos and configs read the same files), as run_merit.sh does
    ( flock 9
      [ -s "$EPIC/outputs/v3/banks/$BANK/${V}_questions.json" ] || (cd "$EPIC" && "$PY" v3/export_bank.py "$BANK") >&2
    ) 9>"$EPIC/outputs/v3/banks/.$BANK.lock"
    RUN=$("$PY" "$M/epic/make_run.py" --config "$M/configs/$CFG.json" --bank "$BANK" --epic "$EPIC" --videos "$VIDEOS" \
        --work "$WORK" ${API_MEDIA_DIR:+--media_dir "$API_MEDIA_DIR"})
}
prep() {
    setup; log "prep"
    # one cut per video at a time: configs with the same media settings share the clips (and their temp names),
    # and the job board runs two configs of a video side by side; the second one finds every clip done
    ( flock 9; "$PY" -m epic_eval.prep_media --run_dir "$RUN" --video "$V" ) 9>"$WORK/locks/prep_$V.lock"
}
qa() {
    setup; log "qa $BANK"
    local rc=0
    "$PY" -m epic_eval.run_eval --run_dir "$RUN" --video "$V" ${API_LIMIT:+--limit "$API_LIMIT"} || rc=$?
    # the answers so far, in the baselines' layout; exit 2 = some questions still have no answer
    "$PY" "$M/epic/from_epic_eval.py" --run "$RUN" --out "$OUT" --arm "$CFG" --bank "$BANK" --video "$V" || rc=$?
    [ "$rc" = 0 ] || { log "qa incomplete (exit $rc): see $RUN/errors/$V.jsonl, then run the same command again"; exit "$rc"; }
}
upload() {
    log "upload"
    "$PY" "$M/../common/hf_results.py" upload --baseline api --memory none --arm "$CFG" --video "$V" --dir "$OUT/$CFG/$V"
}
case $STAGE in
    prep) prep ;;
    qa) prep; qa ;;   # the clips first (skipped when cut already): a job-board task is just `qa`
    upload) upload ;;
    score) setup; "$PY" -m epic_eval.score --run_dir "$RUN" ;;
    all) prep; qa ;;
    *) echo "unknown stage $STAGE (prep, qa, upload, score, all)"; exit 1 ;;
esac
log "$STAGE done"
