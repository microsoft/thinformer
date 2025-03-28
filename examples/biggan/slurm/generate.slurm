#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Script to generate images from BigGAN using a given attention method

OUTPUT_PATH=$1
METHOD=$2

# Run thinformer
for g in 0 1 2 3 4
do
    echo "Running thinformer with g=$g"
    python compress_bigGAN_script.py --num_classes 1000 --data_per_class 5 --beta 0.5 --attention $METHOD -op "${OUTPUT_PATH}-g${g}" --g $g
done