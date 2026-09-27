#!/bin/bash
#SBATCH --job-name=smoke-qwen38
#SBATCH --account=abhinav2-prj-cmsc      # cmsc by default; aac only as last resort
#SBATCH --partition=gpu
#SBATCH --qos=normal                     # MUST be normal for GPU — scavenger is rejected for GPU on Zaratan
#SBATCH --gres=gpu:a100:2                # 27B bf16 is ~54 GB of weights: does not fit one 40 GB A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/vatsalag/scratch/morph_graph/output/logs/smoke_qwen38_%j.out
#SBATCH --error=/home/vatsalag/scratch/morph_graph/output/err/smoke_qwen38_%j.err

# Usage (from ~/scratch/morph_graph; extra args go to the smoke script):
#   sbatch epic_eval/backends/smoke_qwen38.slurm
#   sbatch epic_eval/backends/smoke_qwen38.slurm --attn sdpa
#   sbatch epic_eval/backends/smoke_qwen38.slurm --video /path/clip.mp4

set -eo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID} restarts=${SLURM_RESTART_COUNT:-0}"
source /etc/profile.d/modules.sh
module load apptainer 2>/dev/null || true   # no-op if apptainer is already on PATH
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Built from proj_context/setup/morph_graphs.def (CUDA 12.6, torch 2.8, flash-attn, fla, ffmpeg all inside).
# Override with: SIF=/path/to/other.sif sbatch ...
SIF=${SIF:-/scratch/zt1/project/abhinav2-prj/user/vatsalag/morph_graph/morph_graphs.sif}
# Pure-python packages missing from the image (e.g. ezcolorlog) live here so we don't
# need a rebuild. One-time, from a login node:
#   apptainer exec --cleanenv --bind /scratch "$SIF" pip install --target "$PYEXTRA" ezcolorlog
PYEXTRA=${PYEXTRA:-/scratch/zt1/project/abhinav2-prj/user/vatsalag/pyextra_morph_graphs}
cd /home/vatsalag/scratch/morph_graph
export PYTHONUNBUFFERED=1

# --nv: expose host GPUs/driver. --cleanenv: keep host module/conda compiler vars out,
# or Triton's runtime compile breaks (see proj_context/setup/container_setup.md).
# /scratch is bound because ~/scratch symlinks into it (repo, model_hub, datasets).
apptainer exec --nv --cleanenv --bind /scratch --env PYTHONPATH="$PYEXTRA" "$SIF" \
    python epic_eval/backends/smoke_qwen38.py "$@"
