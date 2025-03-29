#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:a6000:1

# Script to generate images using BigGAN (biggan-deep-512) using a given attention method

OUTPUT_PATH=$1

# # Run thinformer for multiple values of g
# for g in 0 1 2 3 4
# do
#     echo "Running thinformer with g=$g"
#     python compress_bigGAN_script.py --num_classes 1000 --data_per_class 5 --beta 0.5 --attention $METHOD -op "${OUTPUT_PATH}-g${g}" --g $g
# done

METHODS=(
    # exact
    # performer
    reformer
    # kdeformer
    # sblocal
    # thinformer
)
for method in "${METHODS[@]}"
do
    CMD="python compress_bigGAN_script.py --num_classes 1000 --data_per_class 5 --beta 0.5 --attention $method -op $OUTPUT_PATH"
    echo $CMD
    eval $CMD
done