#!/bin/bash
#SBATCH -J longbench-e                       # Job name
#SBATCH -o logs/longbench-e_%A_%a.out        # output file (%A = job ID, %a = array task ID)
#SBATCH -e logs/longbench-e_%A_%a.err        # error log file (%A = job ID, %a = array task ID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH --time=12:00:00                      # time limit
#SBATCH --requeue                            # requeue if preempted

# Script to run longbench-e evaluation with job array
# Usage: ./launch-longbench-e.sh <output_dir> <partition>
#   partition: SLURM partition to submit to
#
# This script can run in two modes:
# 1. Submission mode (no SLURM_ARRAY_TASK_ID): builds commands, writes to file, submits job array
# 2. Execution mode (SLURM_ARRAY_TASK_ID set): reads and executes command for this task

# Check if running as SLURM array task or in submission mode
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # ===== SUBMISSION MODE =====
    echo "Running in submission mode..."

    OUTPUT_DIR=$1
    PARTITION=$2  # SLURM partition

    if [ -z "$OUTPUT_DIR" ] || [ -z "$PARTITION" ]; then
        echo "Error: OUTPUT_DIR and partition must be provided"
        echo "Usage: ./launch-longbench-e.sh <output_dir> <partition>"
        exit 1
    fi

    # Configuration
    dataset="longbench-e"
    model="meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Data directories for longbench-e
    data_dirs=(
        'hotpotqa_e'
        'trec_e'
        'triviaqa_e'
    )

    # These datasets score better without the chat template for Llama-3.1-8B-Instruct
    DATASETS_WITHOUT_CHAT_TEMPLATE=(
        'lcc_e'
        'repobench-p_e'
        'samsum_e'
        'trec_e'
    )

    # Express attention sink / sliding-window sizes (constant for this experiment)
    n_first=32
    n_last=32

    # Build array of all commands
    CMDS=()
    for data_dir in "${data_dirs[@]}"; do
        # Some datasets work better without the chat template
        if [[ " ${DATASETS_WITHOUT_CHAT_TEMPLATE[*]} " =~ " ${data_dir} " ]]; then
            disable_chat_template="True"
        else
            disable_chat_template="False"
        fi
        # Press methods to evaluate - map data_dir to specific press
        # Always include no_press as baseline
        case "$data_dir" in
            hotpotqa_e)
                press_names=('no_press' 'pyramidkv')
                ;;
            triviaqa_e)
                press_names=('no_press' 'streaming_llm')
                ;;
            trec_e)
                press_names=('no_press' 'snapkv')
                ;;
            *)
                # Default: just no_press for unmapped data_dirs
                press_names=('no_press')
                ;;
        esac
        for press_name in "${press_names[@]}"; do
            # Enable PyramidKV's causal_lower_right mask only when running pyramidkv
            if [ "$press_name" = "pyramidkv" ]; then
                use_pyramidkv="True"
            else
                use_pyramidkv="False"
            fi

            # Set target_size and seed based on press_name
            if [ "$press_name" = "no_press" ]; then
                target_sizes=(0)
                seeds=(1)
            else
                # For other methods (pyramidkv, random, snapkv, streaming_llm)
                n_first=32
                n_last=32
                first_plus_last=$((n_first+n_last))
                target_sizes=(
                    $((first_plus_last+32768))
                    $((first_plus_last+16384))
                    $((first_plus_last+8192))
                    $((first_plus_last+4096))
                    $((first_plus_last+2048))
                    $((first_plus_last+1024))
                    $((first_plus_last+512))
                    $((first_plus_last+256))
                )
                seeds=(1)
            fi

            # h values: 9, and None (no --h flag)
            h_values=(
                9
                ""
            )

            for target_size in "${target_sizes[@]}"; do
                for seed in "${seeds[@]}"; do
                    for h in "${h_values[@]}"; do
                        CMD="python runtime.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press_name --target_size $target_size --output_dir $OUTPUT_DIR --seed $seed --do_sample False"
                        # Accelerating KV cache compression experiment uses the Express-prefill attention
                        CMD="$CMD --use_express_prefill True --n_first $n_first --n_last $n_last"
                        CMD="$CMD --disable_chat_template $disable_chat_template --use_pyramidkv $use_pyramidkv"
                        if [ -n "$h" ]; then
                            CMD="$CMD --h $h"
                        fi
                        CMDS+=("$CMD")
                    done
                done
            done
        done
    done

    # Total number of commands
    NUM_JOBS=${#CMDS[@]}
    MAX_CONCURRENT_JOBS=40
    echo "Number of jobs to submit: ${NUM_JOBS}"

    # Write all commands to a file in the current directory (shared filesystem)
    COMMAND_FILE="$(pwd)/longbench_e_commands_$$.txt"
    echo "Writing commands to file: $COMMAND_FILE"
    printf '%s\n' "${CMDS[@]}" > "$COMMAND_FILE"

    # Submit this script as a job array
    echo "Submitting job array with ${NUM_JOBS} tasks"
    JOB_ID=$(sbatch -p "$PARTITION" -n 4 --mem=64GB --gres=gpu:nvidia_rtx_a6000:1 --get-user-env --array=1-${NUM_JOBS}%${MAX_CONCURRENT_JOBS} "$0" "$OUTPUT_DIR" "$COMMAND_FILE" | awk '{print $4}')
    echo "Submitted job array: $JOB_ID"
    echo "Command file created: $COMMAND_FILE"
    echo ""
    echo "Note: The command file is NOT automatically cleaned up"
    echo "To clean it up manually later: rm $COMMAND_FILE"

else
    # ===== EXECUTION MODE =====
    # Running as a SLURM array task

    # Activate the conda environment (project now uses a conda env, not `uv run`)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate kvcache

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

    # All runtime configuration (chat template, pyramidkv, express prefill,
    # sink/window sizes) is passed as explicit CLI flags baked into $CMD in
    # submission mode, so no environment setup is needed here.
    eval $CMD
fi
