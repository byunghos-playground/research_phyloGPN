#!/bin/bash --login
# scripts/train_naive.sbatch
#
# 모델 B (Naive supervised) 훈련 SLURM 스크립트.
#
# 제출 방법:
#   sbatch scripts/train_naive.sbatch
#
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT}

python train_naive.py \
    --train_dir   data/train \
    --valid_dir   data/valid \
    --out_dir     checkpoints/naive \
    --batch_size  4 \
    --epochs      20 \
    --lr          1e-4 \
    --num_workers 4

echo "=== Naive 훈련 완료 ==="
