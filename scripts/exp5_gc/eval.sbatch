#!/bin/bash --login
# scripts/exp5_gc/eval.sbatch
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp5_gc/eval_%j.out
#SBATCH --error=logs/exp5_gc/eval_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
module purge && module load Miniforge3 && conda activate phylogpn-env
cd ${PROJECT} && mkdir -p logs/exp5_gc results/exp5_gc

for MODEL in f81 f81_supervised naive; do
    python evaluate_gc.py \
        --checkpoint  checkpoints/exp5_gc/${MODEL}/best.pt \
        --split_json  checkpoints/exp5_gc/f81/split.json \
        --model_name  exp5_${MODEL} \
        --out_dir     results/exp5_gc \
        --batch_size  8 --stride 481 --block_length 10000 --num_workers 4
done

echo "=== [Exp5] 평가 완료 ==="
