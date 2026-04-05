#!/bin/bash --login
# scripts/exp3_gc/eval.sbatch
#
# [Exp3: gc] F81 / F81-supervised / Naive 모델 평가.
#
# 제출 방법:
#   sbatch scripts/exp3_gc/eval.sbatch
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp3_gc/eval_%j.out
#SBATCH --error=logs/exp3_gc/eval_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT} && mkdir -p logs/exp3_gc results/exp3_gc

for MODEL in f81 f81_supervised naive; do
    python evaluate_gc.py \
        --checkpoint  checkpoints/exp3_gc/${MODEL}/best.pt \
        --split_json  checkpoints/exp3_gc/f81/split.json \
        --model_name  exp3_${MODEL} \
        --out_dir     results/exp3_gc \
        --batch_size  8 --stride 481 --block_length 10000 --num_workers 4
done

echo "=== [Exp3] 평가 완료 ==="
