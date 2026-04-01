#!/bin/bash --login
# scripts/exp2_baseline_r/train_f81_supervised.sbatch
#
# [Exp2: baseline_r] 모델 F81 Supervised 훈련.
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=logs/exp2_baseline_r/train_f81_supervised_%j.out
#SBATCH --error=logs/exp2_baseline_r/train_f81_supervised_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-env

cd ${PROJECT}
mkdir -p logs/exp2_baseline_r

python train_f81_supervised.py \
    --data_dir    data/exp2_baseline_r/processed \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --tree_path   data/trees/241-mammalian-2020v2.1.nh.txt \
    --out_dir     checkpoints/exp2_baseline_r/f81_supervised \
    --batch_size  8 \
    --epochs      20 \
    --lr          1e-4 \
    --num_workers 4

echo "=== [Exp2] F81 Supervised 훈련 완료 ==="
