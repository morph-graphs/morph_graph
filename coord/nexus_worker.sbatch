#!/bin/bash
# One person's job-board worker on Nexus: a small CPU job that keeps claiming tasks for POOL from the board and sends
# each one to every queue in QUEUES that fits its arm; the copy that starts first runs it, the others are cancelled
# (worker.py --queues, coord/slurm_guard.sh). One worker per person, whatever the number of queues.
#
# Once (W is your worker folder, on /fs so the GPU jobs can write their logs and exit codes there):
#   W=/fs/cfar-projects/mem_open_vocab/morph_graphs/$USER/board_worker; mkdir -p $W/logs
#   git clone git@github.com:morph-graphs/morph_graph.git $W/mg      # main; `git -C $W/mg pull` to update
#   (umask 077; echo hf_... > ~/.config/hf_write_token)              # your HF token: write to epic-morph-jobs and -results
# Start (and again after 2 days, when this job's time runs out: running tasks keep going, the new worker picks them up):
#   sbatch --output=$W/logs/%x-%j.out --export=ALL,POOL=vatsal,W=$W,SLOTS=8 $W/mg/coord/nexus_worker.sbatch
# Paid-API tasks (MERIT / WorldMM g5m*, MAGIC oss120*: BASELINES="merit worldmm magic") are taken only with an
# OpenRouter key: one bare line in ~/.config/openrouter_key, mode 600. It is read into this job's environment below
# and reaches each task's job through sbatch's exported environment, never a command line, a script or the board.
# Optional: QUEUES (default $W/mg/coord/nexus_queues.json; copy it to drop or add queues), BASELINES (default merit),
# PY (a Python with huggingface_hub). Check it with `tail -f $W/logs/board-worker-<id>.out` and `squeue -u $USER`.
#SBATCH --job-name=board-worker
#SBATCH --partition=vulcan-cpu
#SBATCH --qos=vulcan-cpu
#SBATCH --account=vulcan-abhinav
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${POOL:?set POOL, your pool name on the board (e.g. vatsal)}" "${W:?set W, your worker folder}"
P=/fs/cfar-projects/mem_open_vocab
QUEUES=${QUEUES:-$W/mg/coord/nexus_queues.json}
PY=${PY:-$P/envs/magic-client/bin/python}           # any venv with huggingface_hub >= 0.30 (the worker needs nothing else)
TOKEN_FILE=${TOKEN_FILE:-$HOME/.config/hf_write_token}
[ "$(stat -c %a "$TOKEN_FILE")" = 600 ] || { echo "$TOKEN_FILE must exist with mode 600"; exit 1; }
export HF_TOKEN=$(cat "$TOKEN_FILE")
KEY_FILE=${OPENROUTER_KEY_FILE:-$HOME/.config/openrouter_key}   # optional: without it the worker leaves the keyed tasks
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
    [ "$(stat -c %a "$KEY_FILE")" = 600 ] || { echo "$KEY_FILE must have mode 600"; exit 1; }
    export OPENROUTER_API_KEY=$(cat "$KEY_FILE")
fi
unset OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY              # no other provider's key in this worker's jobs
echo "OpenRouter key: $([ -n "${OPENROUTER_API_KEY:-}" ] && echo set || echo "not set (keyed tasks are left)")"
export HF_HOME=$W/hf_home                           # the board's and results' downloads (this worker's own cache)
export MERIT_HF_CACHE=$P/hf_cache                   # the models, shared, mounted into the container
export MERIT_SIF_DIR=$P/envs/sif                    # <baseline>_<arm>.sif per task
export MERIT_WORK=$P/merit_work                     # clips, frames, runs: one group-writable folder for every worker on Nexus
export EPIC_MORPH_DIR=${EPIC_MORPH_DIR:-$P/morph_graphs/anubhav/epic-morph}   # the banks qa tasks answer
export HDEPIC_VIDEOS=$P/datasets/HD-EPIC/videos
cd "$W/mg"
# SLOTS = how many tasks at once (each has one copy pending per fitting queue until one starts)
exec "$PY" coord/worker.py --pool "$POOL" --launcher slurm --runtime apptainer --queues "$QUEUES" \
    --slots $(seq 1 "${SLOTS:-4}") --baselines ${BASELINES:-merit} --every 120 --state "$W/state-$POOL"
