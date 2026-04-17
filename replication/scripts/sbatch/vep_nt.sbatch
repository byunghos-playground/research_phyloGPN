#!/bin/bash --login
# replication/scripts/sbatch/vep_nt.sbatch
#
# Nucleotide Transformer (2.5B) VEP — ClinVar & OMIM
# 모델이 크므로 메모리 여유 있게 요청
#
# 제출:
#   sbatch replication/scripts/sbatch/vep_nt.sbatch
#
#SBATCH --job-name=vep_nt
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=[nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=replication/logs/vep_nt_%j.out
#SBATCH --error=replication/logs/vep_nt_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-replication

cd ${PROJECT}
mkdir -p replication/logs replication/results/preds/clinvar replication/results/preds/omim

GENOME=replication/data/raw/hg38/hg38.fa

echo "=== [NT 2.5B] ClinVar ==="
python replication/scripts/vep/run_vep_nt.py \
    --variants  replication/data/processed/clinvar.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/clinvar/nt.parquet \
    --batch_size 16 \
    --device    cuda

echo "=== [NT 2.5B] OMIM ==="
python replication/scripts/vep/run_vep_nt.py \
    --variants  replication/data/processed/omim.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/omim/nt.parquet \
    --batch_size 16 \
    --device    cuda

echo "=== NT VEP 완료 ==="
