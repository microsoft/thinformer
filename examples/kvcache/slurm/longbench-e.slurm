#!/bin/bash
#SBATCH -J longbench-e                       # Job name
#SBATCH -o logs/longbench-e_%j.out           # output file (%j = job ID)
#SBATCH -e logs/longbench-e_%j.err           # error log file (%j = job ID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the user's login environment
#SBATCH --mem=64GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:nvidia_rtx_a6000:1        # GPU requested
#SBATCH --time=infinite                      # no time limit (long sequential sweep)
#SBATCH --requeue                            # requeue if preempted

# Single SLURM job that runs the full longbench-e sweep for ONE data_dir
# sequentially (no job array, no nested sbatch). Submit it directly, passing the
# partition on the command line:
#
#   sbatch -p <partition> slurm/longbench-e.slurm <output_dir> <data_dir>
#
# Run from examples/kvcache/ so that runtime.py and logs/ resolve correctly.

OUTPUT_DIR=$1
DATA_DIR=$2

if [ -z "$OUTPUT_DIR" ] || [ -z "$DATA_DIR" ]; then
    echo "Error: output_dir and data_dir must be provided"
    echo "Usage: sbatch -p <partition> slurm/longbench-e.slurm <output_dir> <data_dir>"
    exit 1
fi

# Activate the conda environment (project uses a conda env, not `uv run`)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvcache

# Configuration
dataset="longbench-e"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Express attention sink / sliding-window sizes (constant for this experiment)
n_first=32
n_last=32

# Map data_dir -> press method(s) to evaluate (always include no_press baseline)
case "$DATA_DIR" in
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

# These tasks work better without chat template for Llama-3.1-8B-Instruct
DATASETS_WITHOUT_CHAT_TEMPLATE=(
    'lcc_e'
    'repobench-p_e'
    'samsum_e'
    'trec_e'
)
if [[ " ${DATASETS_WITHOUT_CHAT_TEMPLATE[*]} " =~ " ${DATA_DIR} " ]]; then
    disable_chat_template="True"
else
    disable_chat_template="False"
fi
echo "Data dir: $DATA_DIR (disable_chat_template: $disable_chat_template)"

# h values: 9, and None (no --h flag)
h_values=(
    9
    ""
)
seed=1

FAILED_CMDS=()
for press_name in "${press_names[@]}"; do
    # Set target_sizes based on press_name
    if [ "$press_name" = "no_press" ]; then
        target_sizes=(0)
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
    fi

    # Enable PyramidKV's causal_lower_right mask only when running pyramidkv
    if [ "$press_name" = "pyramidkv" ]; then
        use_pyramidkv="True"
    else
        use_pyramidkv="False"
    fi
    echo "Press name: $press_name (use_pyramidkv: $use_pyramidkv)"

    for target_size in "${target_sizes[@]}"; do
        for h in "${h_values[@]}"; do
            CMD="python runtime.py --dataset $dataset --data_dir $DATA_DIR --model $model --press_name $press_name --target_size $target_size --output_dir $OUTPUT_DIR --seed $seed --do_sample False"
            # Accelerating KV cache compression experiment uses the Express-prefill attention
            CMD="$CMD --use_express_prefill True --n_first $n_first --n_last $n_last"
            CMD="$CMD --disable_chat_template $disable_chat_template --use_pyramidkv $use_pyramidkv"
            if [ -n "$h" ]; then
                CMD="$CMD --h $h"
            fi

            echo "========================================="
            echo "Command: $CMD"
            echo "========================================="
            eval "$CMD"
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "FAILED with exit code $rc: $CMD"
                FAILED_CMDS+=("$CMD")
            fi
        done
    done
done

echo "========================================="
echo "Sweep complete for data_dir=$DATA_DIR."
if [ ${#FAILED_CMDS[@]} -eq 0 ]; then
    echo "All commands succeeded."
else
    echo "${#FAILED_CMDS[@]} command(s) failed:"
    printf '  %s\n' "${FAILED_CMDS[@]}"
    exit 1
fi
