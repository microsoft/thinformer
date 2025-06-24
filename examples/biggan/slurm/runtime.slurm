#!/bin/bash
#SBATCH --job-name=runtime
#SBATCH --output=logs/runtime_%j.out
#SBATCH --error=logs/runtime_%j.err
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=100GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:a6000:1

# Script to compute runtimes of BigGAN using a given attention method

OUTPUT_PATH=$1

# # Run thinformer
# for g in 0 1 2 3 4
# do
#     echo "Running thinformer with g=$g"
#     python runtime.py --attention thinformer -g $g -op "${OUTPUT_PATH}-g${g}"
# done

METHODS=(
    exact
    performer
    reformer
    kdeformer
    sblocal
    thinformer
)
for method in "${METHODS[@]}"
do
    CMD="python runtime.py --attention $method -op $OUTPUT_PATH -n 10"
    echo $CMD
    eval $CMD
done