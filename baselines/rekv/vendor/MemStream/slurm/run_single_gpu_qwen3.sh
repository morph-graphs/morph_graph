#!/bin/bash

# Qwen3-VL variant of run_single_gpu.sh.
# Differences from that script:
#   - conda env memstream_qwen3_cu126 + cuda/12.6.3 (Qwen3-VL needs transformers >= 4.57;
#     the `memstream` env is far older and cannot import Qwen3VLForConditionalGeneration)
#   - defaults to qwen3_vl_8b on lvbench

#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=memstream_qwen3
#SBATCH --output=output/logs/memstream_qwen3_%j.log
#SBATCH --error=output/err/memstream_qwen3_%j.err

# Check if we're running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    IS_SLURM=true
    export PYTHONBREAKPOINT=0 # disable ipdb
    export ENABLE_LOGGING=false # disable logging
else
    echo "Running as regular bash script"
    IS_SLURM=false
    export ENABLE_LOGGING=false # disable logging
fi

show_usage() {
    echo "Usage:"
    echo "  As SLURM job: sbatch $0 [model] [dataset] [additional_args...]"
    echo "  As bash script: $0 [model] [dataset] [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 qwen3_vl_8b lvbench"
    echo "  $0 qwen3_vl_8b lvbench --encode_chunk_size 32 --resume"
    echo "  $0 qwen3_vl_8b lvbench --num_chunks 52 --chunk_idx 34 --gpus_per_chunk 1 --resume"
    echo ""
    echo "Supported models: qwen3_vl_8b"
    echo "Supported datasets: qaego4d egoschema cgbench cgbench_clue_grounded mlvu activitynet_qa rvs_ego rvs_movie lvbench"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Set up error handling
set -e

# Load necessary modules or environment
echo "Loading modules and environment..."

# Only load modules if we're on a cluster (SLURM environment)
if [[ $IS_SLURM == true ]]; then
    module load cuda/12.6.3
fi

# Create output directories if they don't exist
mkdir -p output/logs output/err

# Get number of GPUs
if command -v nvidia-smi &> /dev/null; then
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $num_gpus GPUs"
else
    echo "Warning: nvidia-smi not found. Setting num_gpus=1"
    num_gpus=1
fi

# Parse arguments with defaults
model=${1:-"qwen3_vl_8b"}
dataset=${2:-"lvbench"}

echo "Configuration:"
echo "  Model: $model"
echo "  Dataset: $dataset"
echo "  Total GPUs available: $num_gpus"

# Shift model and dataset; remaining args pass through to run_eval.py
if [ $# -ge 2 ]; then
    shift 2
elif [ $# -ge 1 ]; then
    shift 1
fi

# Drop empty args (e.g. from accidental "\\   " in a continued command line)
extra_args=()
for arg in "$@"; do
    if [ -n "$arg" ]; then
        extra_args+=("$arg")
    fi
done

# Activate conda environment
if [[ -f "/fs/cfar-projects/mem_open_vocab/miniconda3/bin/activate" ]]; then
    source /fs/cfar-projects/mem_open_vocab/miniconda3/bin/activate memstream_qwen3_cu126
    echo "Activated conda environment: memstream_qwen3_cu126"
fi

echo "Starting evaluation..."
echo "Additional arguments: ${extra_args[*]}"

export DECORD_EOF_RETRY_MAX=20480

# Run the actual command
python -m video_qa.run_eval \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    "${extra_args[@]}"

echo "Evaluation completed successfully!"
