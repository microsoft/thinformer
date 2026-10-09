#!/bin/bash
#SBATCH -J single_attn                         # Job name
#SBATCH -o logs/single_attn_%A_%a.out          # output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/single_attn_%A_%a.err          # error log file
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 4                                   # Total number of cores requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=64GB                             # server memory requested (per node)
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-2                            # 3 configurations total

# Script to benchmark all methods in the single-layer setup using job arrays
# NOTE: we time the attention layer inside the chatglm2-6b-32k model,
# as this yields lower numbers overall for thinformer, but weirdly not for hyper-cuda.
#
# Usage:
# sbatch slurm/benchmark_single_attention_no_causal_array.slurm /path/to/output/dir

OUTPUT_DIR=$1

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not specified"
    echo "Usage: sbatch slurm/benchmark_single_attention_no_causal_array.slurm /path/to/output/dir"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Define methods array (with optional g_eps after colon)
METHODS=(
    "exact"
    "thinformer"    # thinformer (original, non-triton)
    "thinformer_triton"  # thinformer_triton
)

# Get method for this task
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

# Use the same conda environment for all configs
CONDA_ENV="prefill"

echo "=== Configuration ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Method: $METHOD"
echo "Conda environment: $CONDA_ENV"
echo "====================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# Build command
# We don't pass a value of -g, meaning g is chosen automatically for a fixed nout = 256
# Iterate over seq_len from 1k to 32k (powers of 2)
SEQ_LENS=(1024 2048 4096 8192 16384 32768)

OVERALL_EXIT_CODE=0
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    # seq_len is encoded in the output filename, so all seq_lens share one output dir
    CMD="python benchmark_single_attention.py -m $METHOD -bs 1 -hs 32 -d 128 -s 1 --seq_len $SEQ_LEN -op $OUTPUT_DIR --force --in_network --causal 0 --log2_target_cache_size 8"

    echo "Executing: $CMD"
    eval $CMD

    EXIT_CODE=$?
    echo "Command finished with exit code: $EXIT_CODE (seq_len=$SEQ_LEN)"
    if [ $EXIT_CODE -ne 0 ]; then
        OVERALL_EXIT_CODE=$EXIT_CODE
    fi
done

exit $OVERALL_EXIT_CODE
