#!/bin/bash
#SBATCH -J math500-time                          # Job name
#SBATCH -o logs/math500_time_%A_%a.out           # output file (%A = job ID, %a = array task ID)
#SBATCH -e logs/math500_time_%A_%a.err           # error log file (%A = job ID, %a = array task ID)
#SBATCH -N 1                                     # Total number of nodes requested
#SBATCH --time=12:00:00                          # time limit
#SBATCH --requeue                                # requeue if preempted

# Script to run math500 runtime evaluation with job array
# Usage: ./launch-math500-time.sh <output_dir>
#
# Cluster-specific SLURM settings are read from environment variables:
#   SLURM_PARTITION  partition to submit to (default: cluster default)
#   SLURM_GRES       GPU request (default: gpu:nvidia_rtx_a6000:1)
#   SLURM_EXCLUDE    comma-separated nodes to exclude (default: none)
# e.g. SLURM_PARTITION=kilian SLURM_EXCLUDE=nikola-compute-18 ./launch-math500-time.sh out
#
# This script can run in two modes:
# 1. Submission mode (no SLURM_ARRAY_TASK_ID): builds commands, writes to file, submits job array
# 2. Execution mode (SLURM_ARRAY_TASK_ID set): reads and executes command for this task

# Check if running as SLURM array task or in submission mode
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # ===== SUBMISSION MODE =====
    echo "Running in submission mode..."

    OUTPUT_DIR=$1

    if [ -z "$OUTPUT_DIR" ]; then
        echo "Error: OUTPUT_DIR not provided"
        echo "Usage: ./launch-math500-time.sh <output_dir>"
        exit 1
    fi

    # Configuration
    dataset="math500"
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    batch_size=5
    max_new_tokens=32768
    seed=1

    # Cluster-specific SLURM options (override via environment variables)
    SBATCH_OPTS=(-n 4 --mem=64GB --gres="${SLURM_GRES:-gpu:nvidia_rtx_a6000:1}" --get-user-env)
    if [ -n "$SLURM_PARTITION" ]; then
        SBATCH_OPTS+=(-p "$SLURM_PARTITION")
    fi
    if [ -n "$SLURM_EXCLUDE" ]; then
        SBATCH_OPTS+=(--exclude="$SLURM_EXCLUDE")
    fi

    press_names=(
        'no_press'
        'express_new_32_32'
        'balancekv_express'
        'decoding_expected_attention'
        'decoding_snapkv'
        'decoding_keydiff'
        'decoding_streaming_llm'
        'decoding_knorm'
        'decoding_balancekv'
    )

    MAX_CONCURRENT_JOBS=8
    TOTAL_JOBS=0
    COMMAND_FILES=()

    # Submit one job array per press_name to keep array sizes well under
    # SLURM's MaxArraySize limit (typically 1001).
    for press_name in "${press_names[@]}"; do
        # if press_name is no_press, then compression_ratio is 0
        if [ "$press_name" = "no_press" ]; then
            target_sizes=(0)
        elif [ "$press_name" = "express_new_32_32" ] || [ "$press_name" = "balancekv_express" ]; then
            target_sizes=(
                12
                11
                10
                9
                8
            )
        elif [ "$press_name" = "decoding_streaming_llm" ] || [ "$press_name" = "decoding_knorm" ]; then
            target_sizes=(
                1500
                2000
                3000
                4000
            )
        else
            target_sizes=(
                500
                1000
                1500
                2000
                3000
                4000
            )
        fi

        # Build commands for just this press
        CMDS=()
        for target_size in "${target_sizes[@]}"; do
            for batch_number in {1..100}; do
                # Note: each batch is 5 samples
                CMD="python runtime.py --dataset $dataset --model $model --press_name $press_name --target_size $target_size --output_dir $OUTPUT_DIR --seed $seed --batch_number $batch_number --batch_size $batch_size --max_new_tokens $max_new_tokens"
                # decoding_balancekv must compress at exactly 2*target_size (compression ratio 0.5)
                if [ "$press_name" = "decoding_balancekv" ]; then
                    CMD="$CMD --compression_interval $target_size"
                fi
                CMDS+=("$CMD")
            done
        done

        NUM_JOBS=${#CMDS[@]}
        TOTAL_JOBS=$((TOTAL_JOBS + NUM_JOBS))

        # Write this press's commands to its own file (shared filesystem)
        COMMAND_FILE="$(pwd)/math500_runtime_commands_${press_name}_$$.txt"
        printf '%s\n' "${CMDS[@]}" > "$COMMAND_FILE"
        COMMAND_FILES+=("$COMMAND_FILE")

        echo "Submitting job array for press '${press_name}': ${NUM_JOBS} tasks (file: $COMMAND_FILE)"
        JOB_ID=$(sbatch "${SBATCH_OPTS[@]}" --array=1-${NUM_JOBS} -J "math500-time-${press_name}" "$0" "$OUTPUT_DIR" "$COMMAND_FILE" | awk '{print $4}')
        echo "  Submitted job array: $JOB_ID"
    done

    echo ""
    echo "Total jobs submitted across all presses: $TOTAL_JOBS"
    echo "Command files created:"
    for f in "${COMMAND_FILES[@]}"; do
        echo "  $f"
    done
    echo ""
    echo "Note: The command files are NOT automatically cleaned up"
    echo "To clean them up manually later: rm math500_runtime_commands_*_$$.txt"

else
    # ===== EXECUTION MODE =====
    # Running as a SLURM array task
    OUTPUT_DIR=$1
    COMMAND_FILE=$2

    if [ -z "$COMMAND_FILE" ] || [ ! -f "$COMMAND_FILE" ]; then
        echo "Error: Command file not provided or does not exist: $COMMAND_FILE"
        exit 1
    fi

    # Read the command for this array task
    CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")

    echo "========================================="
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Command: $CMD"
    echo "========================================="

    eval $CMD
fi
