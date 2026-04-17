#!/bin/bash --login
# replication/scripts/sbatch/vep_phylogpn.sbatch
#
# PhyloGPN VEP — ClinVar & OMIM
#
# 제출:
#   sbatch replication/scripts/sbatch/vep_phylogpn.sbatch
#
#SBATCH --job-name=vep_phylogpn
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=[intel18|(amr|acm)|nvf|nal|nif|agg|agx|(nfh|neh)]
#SBATCH -A liulab
#SBATCH --output=replication/logs/vep_phylogpn_%j.out
#SBATCH --error=replication/logs/vep_phylogpn_%j.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

module purge
module load Miniforge3
conda activate phylogpn-replication

cd ${PROJECT}
mkdir -p replication/logs replication/results/preds/clinvar replication/results/preds/omim

GENOME=replication/data/raw/hg38/hg38.fa

echo "=== [PhyloGPN] ClinVar ==="
python replication/scripts/vep/run_vep_phylogpn.py \
    --variants  replication/data/processed/clinvar.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/clinvar/phylogpn.parquet \
    --batch_size 64 \
    --device    cuda

echo "=== [PhyloGPN] OMIM ==="
python replication/scripts/vep/run_vep_phylogpn.py \
    --variants  replication/data/processed/omim.parquet \
    --genome    ${GENOME} \
    --out       replication/results/preds/omim/phylogpn.parquet \
    --batch_size 64 \
    --device    cuda

echo "=== PhyloGPN VEP 완료 ==="
