#!/bin/bash

# Qwen3-VL on H100/H200 nodes. Copy of run_single_gpu_array_h100.sh with the Qwen3-VL
# fixes from run_single_gpu_array_a5000_scav_qwen3_4gpu.sh folded in.
#
# Why this exists: the 448-512 tpf arm (~480 tok/frame) OOMs the *vision tower* on A5000s.
# get_video_features() in model/qwen3_vl_rekv.py:266 pushes encode_chunk_size x ~480 visual
# tokens through in one shot on a single device, and 23.5 GiB is not enough. Sharding the LLM
# over 4 A5000s does not help -- the encoder peak lands on GPU 0 regardless. One H200 (141 GiB)
# clears it outright, so this script goes back to 1 GPU per chunk.
#
# Differences from run_single_gpu_array_h100.sh:
#   - conda env memstream_qwen3_cu126 + cuda/12.6.3 unconditionally (Qwen3-VL needs
#     transformers >= 4.57; the `memstream` env cannot import Qwen3VLForConditionalGeneration)
#   - 52 array tasks, not 100: LVBench has 103 videos and video_qa/base.py get_chunk() raises
#     IndexError on empty slices. Change --array and NUM_ARRAY_JOBS together; they must match.
#   - --gres=gpu:h200-sxm:1 instead of a bare gpu:1 + --nodelist=cml35. The gres *type* is what
#     pins this to Hopper; it also keeps tasks off cml38 (b300-sxm is sm_100 and the cu126 env
#     has no kernels for it). For H100 instead:
#         sbatch --gres=gpu:h100-sxm:1 slurm/run_single_gpu_array_h200_qwen3.sh ...
#     (cml33, 4x h100-sxm; check `sinfo` first, it has been draining). cml31 carries a single
#     h100-nvl shared with an a100 -- usable but only one task at a time.
#     Non-preemptible alternative on vulcan46 (8x h200-sxm), if scavenger churn is a problem:
#         sbatch --partition=vulcan-ampere --qos=vulcan-medium-h200 --account=vulcan-abhinav \
#                slurm/run_single_gpu_array_h200_qwen3.sh ...
#     cml-scavenger caps a user at 24 gres/gpu, so at 1 GPU/task ~24 chunks run concurrently.
#   - --mem=400G, not 64G. This is HOST ram and it is the real constraint: the offloaded KV
#     cache runs ~147 KB/token, so a 2h LVBench video at 0.5 fps and 480 tok/frame wants a few
#     hundred GB. Two tasks of the previous A5000 run died at the cgroup limit with
#     MaxRSS ~192 GiB. The h200 nodes carry 2 TB, so this is affordable.
#   - RC=$? captured immediately. The original re-read $? after an echo, so every failed chunk
#     reported "exit code 0" and hid the real cause (137 = OOM-killed).
#
# NOTE (resume): video_qa/base.py skips a video only once *all* its questions are answered, and
# encode_video runs before the first question. A video that dies during encoding therefore has
# zero completed questions, is never skipped, and re-crashes at the same point on every resume.
# --resume only makes progress here because the H200 removes the crash; it cannot step over one.

#SBATCH --array=1-52
#SBATCH --qos=cml-scavenger
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
# GPU count: override BOTH together or run_eval will address a device that was not allocated,
#   sbatch --gres=gpu:h200-sxm:1 --export=ALL,GPUS_PER_CHUNK=1 ...
# 1 is usually right: measured peak on the worst LVBench video was 18.8 GiB allocated /
# 20.7 GiB reserved of 139.8 GiB. The 2nd GPU only shards the LLM; the vision tower (the
# thing that OOMs on A5000) runs entirely on GPU 0 regardless. At the 24-gpu/user cap,
# 1 GPU/task doubles concurrency from 12 to 24 tasks.
#SBATCH --gres=gpu:h200-sxm:2
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --job-name=memstream_qwen3_h200_array
#SBATCH --output=output/logs/memstream_qwen3_h200_array_%A_%a.log
#SBATCH --error=output/err/memstream_qwen3_h200_array_%A_%a.err

# Parse arguments with defaults
MODEL_NAME=${1:-"qwen3_vl_8b"}
DATASET=${2:-"lvbench"}
SNAPSHOT_NAME=${3:-""}  # pre-created code snapshot name
# Overridable so one script serves datasets with different video counts. NOTE: the REAL
# chunk count is ceil(len/ceil(len/N)), not N -- LVBench 103 videos @52 -> 52, but CG-Bench
# 1219 @100 -> chunk_size 13 -> only 94 chunks, and array ids 95-100 raise IndexError.
NUM_ARRAY_JOBS=${NUM_ARRAY_JOBS:-52}

# Setup temp directory - use pre-created snapshot if provided
PROJECT_ROOT=$(pwd)

if [ -n "$SNAPSHOT_NAME" ]; then
    TEMP_DIR="$PROJECT_ROOT/.temp_code_${SNAPSHOT_NAME}"
    if [ ! -d "$TEMP_DIR" ]; then
        echo "ERROR: Snapshot directory not found: $TEMP_DIR"
        echo "Please run: ./slurm/setup_code_snapshot.sh $SNAPSHOT_NAME"
        exit 1
    fi
    echo "Using pre-created code snapshot: $SNAPSHOT_NAME"
else
    echo "ERROR: No snapshot name provided!"
    echo "Usage: sbatch slurm/run_single_gpu_array_h200_qwen3.sh [model] [dataset] [snapshot_name] [other_args...]"
    echo "First create a snapshot: ./slurm/setup_code_snapshot.sh [snapshot_name]"
    exit 1
fi

cd "$TEMP_DIR"

# Activate conda environment
if [[ -f "/fs/cfar-projects/mem_open_vocab/miniconda3/bin/activate" ]]; then
    source /fs/cfar-projects/mem_open_vocab/miniconda3/bin/activate memstream_qwen3_cu126
    echo "Activated conda environment: memstream_qwen3_cu126"
fi

export DECORD_EOF_RETRY_MAX=20480

# Long videos can exhaust vm.max_map_count via pinned KV offloads; see
# docs/qwen3_vl_max_map_count_bug.md. Fix: pass --no_pin_memory and/or REKV_DISABLE_PIN=1.
# The A5000 failures showed ~491 MiB reserved-but-unallocated at the moment of the OOM; the
# vision tower allocates in bursts of wildly different shapes and fragments the caching
# allocator. Harmless on a 141 GiB card, but keep it -- it costs nothing.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
module load cuda/12.6.3

# CHUNK_OFFSET lets an array address chunks beyond MaxArraySize (1001 here):
# submit --array=K with CHUNK_OFFSET=N to run chunk_idx K-1+N.
CHUNK_IDX=$((SLURM_ARRAY_TASK_ID - 1 + ${CHUNK_OFFSET:-0}))
NUM_CHUNKS=$NUM_ARRAY_JOBS

echo "Processing chunk $SLURM_ARRAY_TASK_ID of $NUM_CHUNKS (chunk_idx=$CHUNK_IDX)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run the actual command
python -m video_qa.run_eval \
    --num_chunks ${NUM_CHUNKS} \
    --model ${MODEL_NAME} \
    --dataset ${DATASET} \
    --sample_fps 0.5 \
    --retrieve_size 64 \
    --chunk_idx ${CHUNK_IDX} \
    --gpus_per_chunk ${GPUS_PER_CHUNK:-2} \
    "${@:4}"

# Capture the exit code immediately: the original script re-read $? after an echo, so a failed
# chunk always reported "exit code 0" and hid the real cause (e.g. 137 = OOM-killed).
RC=$?

if [ $RC -eq 0 ]; then
    echo "Chunk $CHUNK_IDX completed successfully!"
    echo "SUCCESS: Chunk $CHUNK_IDX completed at $(date)" >> output/logs/success_${SLURM_ARRAY_JOB_ID}.log
else
    echo "ERROR: Chunk $CHUNK_IDX failed with exit code $RC"
    {
        echo "FAILED: Chunk $CHUNK_IDX failed at $(date) with exit code $RC"
        echo "Model: $MODEL_NAME, Dataset: $DATASET"
        echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
        echo "---"
    } >> output/logs/failed_${SLURM_ARRAY_JOB_ID}.log
    exit $RC
fi
