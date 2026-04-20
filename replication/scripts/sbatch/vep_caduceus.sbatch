#!/bin/bash --login
# replication/scripts/sbatch/vep_caduceus.sbatch
#
# Caduceus VEP — ClinVar & OMIM
# RC 추론 2회 실행하므로 일반 masked LM보다 ~2배 소요
#
# 제출:
#   sbatch replication/scripts/sbatch/vep_caduceus.sbatch
#
#SBATCH --job-name=vep_caduceus
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=replication/logs/vep_caduceus_%j.out
#SBATCH --error=replication/logs/vep_caduceus_%j.err

export HF_HOME=/mnt/research/liulab/RA_lee/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/research/liulab/RA_lee/.cache/huggingface

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-replication

cd ${PROJECT}
mkdir -p replication/logs replication/results/preds/clinvar replication/results/preds/omim

GENOME=replication/data/raw/hg38/hg38.fa

echo "=== [Caduceus] ClinVar ==="
python replication/scripts/vep/run_vep_caduceus.py \
    --variants  replication/data/processed/clinvar.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/clinvar/caduceus.parquet \
    --batch_size 32 \
    --device    cuda

echo "=== [Caduceus] OMIM ==="
python replication/scripts/vep/run_vep_caduceus.py \
    --variants  replication/data/processed/omim.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/omim/caduceus.parquet \
    --batch_size 32 \
    --device    cuda

echo "=== Caduceus VEP 완료 ==="
