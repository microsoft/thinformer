#!/bin/bash
#SBATCH -J perplexity                         # Job name
#SBATCH -o logs/perplexity_%A_%a.out          # output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/perplexity_%A_%a.err          # error log file
#SBATCH -N 1                                  # Total number of nodes requested
#SBATCH -n 4                                  # Total number of cores requested
#SBATCH --get-user-env                        # retrieve the users login environment
#SBATCH --mem=64GB                            # server memory requested (per node)
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-42                          # 1 (flash) + 3 methods x 14 num_patch_layers = 43 configurations

# Script to compute perplexity of the model with layers replaced.
# Usage:
# sbatch slurm/benchmark_patch_llm_array.slurm /path/to/output/dir

OUTPUT_DIR=$1

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not specified"
    echo "Usage: sbatch slurm/benchmark_patch_llm_array.slurm /path/to/output/dir"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Express attention sink / sliding-window sizes (passed as CLI flags below)
N_FIRST=32
N_LAST=32
echo "N_FIRST: $N_FIRST"
echo "N_LAST: $N_LAST"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Define methods (with optional g_eps after colon)
# flash is special: doesn't need num_patch_layers (task 0 only)
# Other methods need num_patch_layers sweep
METHODS=(
    "flash"
    "hyper-cuda"
    "thinformer_triton:9"    # Thinformer Express with h=9
    "thinformer_triton:10"    # Thinformer Express with h=10
)

# Define num_patch_layers: 2, 4, 6, ..., 28
NUM_PATCH_LAYERS=(2 4 6 8 10 12 14 16 18 20 22 24 26 28)
NUM_LAYERS=${#NUM_PATCH_LAYERS[@]}

# Task ID layout:
# 0: flash (no num_patch_layers)
# 1-14: hyper-cuda with num_patch_layers 2,4,...,28
# 15-28: thinformer_triton:9 with num_patch_layers 2,4,...,28
# 29-42: thinformer_triton:10 with num_patch_layers 2,4,...,28

# Use the same conda environment for all configs
CONDA_ENV="prefill"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Flash: no num_patch_layers needed
    METHOD="flash"
    H=""
    NUM_PATCH=""
else
    # Other methods: need num_patch_layers
    ADJUSTED_ID=$((SLURM_ARRAY_TASK_ID - 1))
    METHOD_IDX=$((ADJUSTED_ID / NUM_LAYERS + 1))  # +1 to skip flash
    LAYER_IDX=$((ADJUSTED_ID % NUM_LAYERS))

    CONFIG=${METHODS[$METHOD_IDX]}
    NUM_PATCH=${NUM_PATCH_LAYERS[$LAYER_IDX]}

    # Parse method and optional h
    METHOD=$(echo $CONFIG | cut -d':' -f1)
    H=$(echo $CONFIG | cut -d':' -f2 -s)
fi

echo "=== Configuration ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Method: $METHOD"
echo "Num patch layers: $NUM_PATCH"
echo "H: $H"
echo "Conda environment: $CONDA_ENV"
echo "====================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# Build command
CMD="python benchmark_patch_llm.py --attn_method $METHOD -od $OUTPUT_DIR --n_first $N_FIRST --n_last $N_LAST"

if [ -n "$NUM_PATCH" ]; then
    CMD="$CMD --num_patch_layers $NUM_PATCH"
fi

if [ -n "$H" ]; then
    CMD="$CMD --log2_target_cache_size $H"
fi

echo "Executing: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Command finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
