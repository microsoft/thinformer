#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:a6000:1

METHODS=(
    exact
    performer
    reformer
    sblocal
    kdeformer
    thinformer
)
for method in "${METHODS[@]}"
do
    CMD="python demo_generate_images_insu.py --data_per_class 5 --fid --attention $method"
    echo $CMD
    eval $CMD
done