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

cd ${PROJECT} && mkdir -p logs/exp1_baseline results/exp1_baseline

for MODEL in f81 f81_supervised naive; do
    python evaluate.py \
        --checkpoint checkpoints/exp1_baseline/${MODEL}/best.pt \
        --split_json checkpoints/exp1_baseline/f81/split.json \
        --model_name exp1_${MODEL} \
        --out_dir    results/exp1_baseline \
        --batch_size 8 --num_workers 4
done

echo "=== [Exp1] 평가 완료 ==="
