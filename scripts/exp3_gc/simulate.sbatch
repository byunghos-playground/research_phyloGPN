#!/bin/bash --login
# scripts/exp3_gc/simulate.sbatch
#
# [Exp3: gc] GC OU process + F81 시뮬레이션 (rate variation 없음).
# GC(x) ~ OU(μ=0.41, σ=0.07, θ=1/5000), L=10,000 bp per genome.
#
# [규모]
#   --array=0-99  → 100 jobs × 30 genomes = 3,000 genomes
#   genome ID: job_id * 30 + i  (00000 ~ 02999)
#
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH -A liulab

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/exp3_gc/raw
PROCESSED_DIR=${PROJECT}/data/exp3_gc/processed

module purge
module load Miniforge3
conda activate phylogpn-env

mkdir -p ${RAW_DIR} ${PROCESSED_DIR}

GENOMES_PER_JOB=30

for i in $(seq 0 $((GENOMES_PER_JOB - 1))); do
    GLOBAL_ID=$((SLURM_ARRAY_TASK_ID * GENOMES_PER_JOB + i))
    GENOME_ID=$(printf "%05d" ${GLOBAL_ID})
    PREFIX=${RAW_DIR}/genome_${GENOME_ID}

    echo "=== Genome ${GENOME_ID} 시뮬레이션 시작 ==="

    python ${PROJECT}/data/simulate/simulate_exp3_gc.py \
        --tree_path  ${TREE} \
        --L          10000 \
        --out_prefix ${PREFIX} \
        --seed       ${GLOBAL_ID}

    python ${PROJECT}/data/simulate/fasta_to_npz.py \
        --fasta  ${PREFIX}.fasta \
        --pi     ${PREFIX}_pi.txt \
        --out    ${PROCESSED_DIR}/genome_${GENOME_ID}.npz

    echo "=== Genome ${GENOME_ID} 완료 ==="
done
