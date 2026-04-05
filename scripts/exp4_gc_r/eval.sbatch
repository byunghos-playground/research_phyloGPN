#!/bin/bash --login
# scripts/exp4_gc_r/eval.sbatch
#
# [Exp4: gc_r] F81 / F81-supervised / Naive 모델 평가.
#
# 제출 방법:
#   sbatch scripts/exp4_gc_r/eval.sbatch
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp4_gc_r/eval_%j.out
#SBATCH --error=logs/exp4_gc_r/eval_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT} && mkdir -p logs/exp4_gc_r results/exp4_gc_r

for MODEL in f81 f81_supervised naive; do
    python evaluate_gc.py \
        --checkpoint  checkpoints/exp4_gc_r/${MODEL}/best.pt \
        --split_json  checkpoints/exp4_gc_r/f81/split.json \
        --model_name  exp4_${MODEL} \
        --out_dir     results/exp4_gc_r \
        --batch_size  8 --stride 481 --block_length 10000 --num_workers 4
done

echo "=== [Exp4] 평가 완료 ==="
