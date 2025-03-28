#!/bin/bash
#SBATCH --job-name=runtime
#SBATCH --output=logs/runtime_%j.out
#SBATCH --error=logs/runtime_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Script to compute runtimes of BigGAN using a given attention method

OUTPUT_PATH=$1
METHOD=$2

# Run thinformer
for g in 0 1 2 3 4
do
    echo "Running thinformer with g=$g"
    python runtime.py --attention thinformer -g $g -op "${OUTPUT_PATH}-g${g}"
done