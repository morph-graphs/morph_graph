#!/bin/bash
# Evidence-clips videos (baselines/common/evclips_stitch.py), one shard of the plan's files per array task, CPU only,
# inside MERIT's image so the encoder (its ffmpeg) is the one prepare_epic encodes Direct's and MERIT's videos with.
# Skips files already written, so a preempted (requeued) task resumes. After every task: the `manifest` step.
# Sizing (smoke test, 2026-11-15): 16 ffmpeg at once peaked at 7.8 GB RSS; ~9 CPU-s per file.
#
#   OUT=<out dir> REPO=<morph_graph checkout> sbatch --array=0-<N-1> run_evclips_stitch.sbatch   (N = N_SHARDS)
#SBATCH --job-name=evclips-stitch
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.log
set -euo pipefail
: "${OUT:?}" "${REPO:?}"
SIF=${SIF:-/fs/cfar-projects/mem_open_vocab/envs/sif/merit_m8.sif}
VIDEOS=${VIDEOS:-/fs/cfar-projects/mem_open_vocab/datasets/HD-EPIC/videos}
N_SHARDS=${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
JOBS=${JOBS:-$SLURM_CPUS_PER_TASK}
THREADS=${THREADS:-1}
echo "$(date -Is) $(hostname) task ${SLURM_ARRAY_TASK_ID:-0}/$N_SHARDS jobs $JOBS threads $THREADS"
apptainer exec --bind /fs/cfar-projects --bind "$REPO" --bind "$OUT" "$SIF" \
    python3 "$REPO/baselines/common/evclips_stitch.py" stitch --out "$OUT" --video_dir "$VIDEOS" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --n_shards "$N_SHARDS" --jobs "$JOBS" --threads "$THREADS"
