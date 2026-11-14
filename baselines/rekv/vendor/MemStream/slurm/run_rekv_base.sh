# Shell/env
source ~/.zshrc
# Prefer conda hook (works reliably in non-interactive shells)
eval "$(conda shell.zsh hook)" || true
conda activate memstream || source ~/miniconda3/bin/activate memstream

set -euo pipefail

# GPUs
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  echo "Detected $num_gpus GPUs"
else
  echo "Warning: nvidia-smi not found. Setting num_gpus=1"
  num_gpus=1
fi

# Args
model=${1:-"llava_ov_0.5b"}
dataset=${2:-"rvs_ego"}
num_chunks=$num_gpus

echo "Configuration:"
echo "  Model: $model"
echo "  Dataset: $dataset"
echo "  Number of chunks: $num_chunks"
echo "  Total GPUs available: $num_gpus"

case $model in
  "llava_ov_72b")
    required_gpus=$((num_chunks * 4))
    echo "Warning: $model needs 4x 80GB GPUs per chunk (total: ${required_gpus} GPUs)"
    ;;
esac

# Shift consumed args
if [ $# -ge 2 ]; then shift 2
elif [ $# -ge 1 ]; then shift 1
fi

export DECORD_EOF_RETRY_MAX=20480

python -m video_qa.run_eval \
  --num_chunks "$num_chunks" \
  --model "$model" \
  --dataset "$dataset" \
  --sample_fps 0.5 \
  --n_local 15000 \
  --retrieve_size 64 \
  "$@"

echo "Evaluation completed successfully!"

echo "Copying results to hdfs..."
mkdir -p "/mnt/hdfs/vatsal.agarwal/results"
rsync -a --ignore-existing --info=progress2 -h "results/" "/mnt/hdfs/vatsal.agarwal/results/"
echo "Results successfully copied!"
