#!/bin/bash --login
# scripts/exp2_baseline_r/eval.sbatch
#
# [Exp2: baseline_r] F81 / F81-supervised / Naive 모델 평가.
#
# 제출 방법:
#   sbatch scripts/exp2_baseline_r/eval.sbatch
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp2_baseline_r/eval_%j.out
#SBATCH --error=logs/exp2_baseline_r/eval_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT} && mkdir -p logs/exp2_baseline_r results/exp2_baseline_r

for MODEL in f81 f81_supervised naive; do
    python evaluate.py \
        --checkpoint checkpoints/exp2_baseline_r/${MODEL}/best.pt \
        --split_json checkpoints/exp2_baseline_r/f81/split.json \
        --model_name exp2_${MODEL} \
        --out_dir    results/exp2_baseline_r \
        --batch_size 8 --num_workers 4
done

echo "=== [Exp2] 평가 완료 ==="
