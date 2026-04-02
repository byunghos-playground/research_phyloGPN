#!/bin/bash --login
# scripts/exp3_gc/train_f81_supervised_gc.sbatch
#
# [Exp3: gc] 모델 F81 Supervised 훈련 (sliding window).
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp3_gc/train_f81_supervised_%j.out
#SBATCH --error=logs/exp3_gc/train_f81_supervised_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT}
mkdir -p logs/exp3_gc

python train_f81_supervised_gc.py \
    --data_dir    data/exp3_gc/processed \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --tree_path   data/trees/241-mammalian-2020v2.1.nh.txt \
    --out_dir     checkpoints/exp3_gc/f81_supervised \
    --batch_size  8 \
    --epochs      20 \
    --lr          1e-4 \
    --num_workers 4 \
    --stride      481 \
    --block_length 10000

echo "=== [Exp3] F81 Supervised 훈련 완료 ==="
