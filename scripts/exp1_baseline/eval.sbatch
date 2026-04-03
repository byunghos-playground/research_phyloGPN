#!/bin/bash --login
# scripts/exp1_baseline/eval.sbatch
#
# [Exp1: baseline] F81 / F81-supervised / Naive 모델 평가.
#
# 제출 방법:
#   sbatch scripts/exp1_baseline/eval.sbatch
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp1_baseline/eval_%j.out
#SBATCH --error=logs/exp1_baseline/eval_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT}
mkdir -p logs/exp1_baseline results/exp1_baseline

echo "=== [Exp1] F81 평가 ==="
python evaluate.py \
    --checkpoint checkpoints/exp1_baseline/f81/best.pt \
    --split_json checkpoints/exp1_baseline/f81/split.json \
    --model_name exp1_f81 \
    --out_dir    results/exp1_baseline \
    --batch_size 8 \
    --num_workers 4

echo "=== [Exp1] F81-Supervised 평가 ==="
python evaluate.py \
    --checkpoint checkpoints/exp1_baseline/f81_supervised/best.pt \
    --split_json checkpoints/exp1_baseline/f81/split.json \
    --model_name exp1_f81_supervised \
    --out_dir    results/exp1_baseline \
    --batch_size 8 \
    --num_workers 4

echo "=== [Exp1] Naive 평가 ==="
python evaluate.py \
    --checkpoint checkpoints/exp1_baseline/naive/best.pt \
    --split_json checkpoints/exp1_baseline/f81/split.json \
    --model_name exp1_naive \
    --out_dir    results/exp1_baseline \
    --batch_size 8 \
    --num_workers 4

echo "=== [Exp1] 평가 완료 ==="
