#!/bin/bash
#SBATCH -J single_attn                         # Job name
#SBATCH -o logs/single_attn_%A_%a.out          # output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/single_attn_%A_%a.err          # error log file
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 4                                   # Total number of cores requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=32GB                             # server memory requested (per node)
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-3                            # 4 configurations total

# Script to benchmark all methods in the single-layer setup using job arrays
# NOTE: we time the attention layer inside the chatglm2-6b-32k model,
# as this yields lower numbers overall for thinformer, but weirdly not for hyper-cuda.
#
# Usage:
# sbatch slurm/benchmark_single_attention_array.slurm /path/to/output/dir [in_network] [n_first] [n_last]
# in_network: 0 or 1 (default: 1)
# n_first:    attention sink size, --n_first (default: 0).
# n_last:     sliding-window size, --n_last (default: same as n_first).
#             Pass n_first=32 (n_last=32) for the masked prefill (n=32768) experiment.

OUTPUT_DIR=$1
IN_NETWORK=${2:-1}
N_FIRST=${3:-0}
N_LAST=${4:-$N_FIRST}

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not specified"
    echo "Usage: sbatch slurm/benchmark_single_attention_array.slurm /path/to/output/dir [in_network] [n_first] [n_last]"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Attention sink / sliding-window sizes (passed as CLI flags below)
echo "N_FIRST: $N_FIRST"
echo "N_LAST: $N_LAST"
echo "PYTHONPATH: $PYTHONPATH"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Define methods array (with optional g_eps after colon)
METHODS=(
    "exact"
    "hyper-cuda"
    "thinformer_triton:9"  # Thinformer Express with h=9
    "thinformer_triton:10"    # Thinformer Express with h=10
)

# Get method for this task
CONFIG=${METHODS[$SLURM_ARRAY_TASK_ID]}

# Parse method and optional g_eps
METHOD=$(echo $CONFIG | cut -d':' -f1)
H=$(echo $CONFIG | cut -d':' -f2 -s)

# Use the same conda environment for all configs
CONDA_ENV="prefill"

echo "=== Configuration ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Method: $METHOD"
echo "H: $H"
echo "Conda environment: $CONDA_ENV"
echo "In network: $IN_NETWORK"
echo "====================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# Build command
CMD="python benchmark_single_attention.py -m $METHOD -bs 1 -hs 32 -d 128 -s 1 -op $OUTPUT_DIR --force --n_first $N_FIRST --n_last $N_LAST"

if [ "$IN_NETWORK" = "1" ]; then
    CMD="$CMD --in_network --time_all"
fi

if [ -n "$H" ]; then
    CMD="$CMD --log2_target_cache_size $H"
fi

echo "Executing: $CMD"
eval $CMD

EXIT_CODE=$?
echo "Command finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
