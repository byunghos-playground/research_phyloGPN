#!/bin/bash --login
# replication/scripts/sbatch/vep_hyenadna.sbatch
#
# HyenaDNA (large-1m) VEP — ClinVar & OMIM
#
# 제출:
#   sbatch replication/scripts/sbatch/vep_hyenadna.sbatch
#
#SBATCH --job-name=vep_hyenadna
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=replication/logs/vep_hyenadna_%j.out
#SBATCH --error=replication/logs/vep_hyenadna_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-replication

cd ${PROJECT}
mkdir -p replication/logs replication/results/preds/clinvar replication/results/preds/omim

GENOME=replication/data/raw/hg38/hg38.fa

echo "=== [HyenaDNA] ClinVar ==="
python replication/scripts/vep/run_vep_hyenadna.py \
    --variants  replication/data/processed/clinvar.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/clinvar/hyenadna.parquet \
    --batch_size 64 \
    --device    cuda

echo "=== [HyenaDNA] OMIM ==="
python replication/scripts/vep/run_vep_hyenadna.py \
    --variants  replication/data/processed/omim.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/omim/hyenadna.parquet \
    --batch_size 64 \
    --device    cuda

echo "=== HyenaDNA VEP 완료 ==="
