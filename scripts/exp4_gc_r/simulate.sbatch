#!/bin/bash --login
# scripts/exp4_gc_r/simulate.sbatch
#
# [Exp4: gc_r] GC OU process 기반 연속 π 변화 + per-site r ~ Gamma(0.5) 시뮬레이션.
# GC(x) ~ OU(mu=0.41, sigma=0.07, theta=1/100000), π(x) = f(GC(x)).
# r(x) ~ Gamma(shape=0.5) 독립 샘플.
#
# [규모]
#   --array=0-999  → 1,000 jobs × 10 genomes = 10,000 genomes
#
# [전체 파이프라인]
#   1. sbatch --array=0-999 scripts/exp4_gc_r/simulate.sbatch
#   2. sbatch scripts/exp4_gc_r/train_f81_gc.sbatch  (+ supervised, naive)
#
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH -A liulab

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/exp4_gc_r/raw
PROCESSED_DIR=${PROJECT}/data/exp4_gc_r/processed

module purge
module load Miniforge3
conda activate phylogpn-env

mkdir -p ${RAW_DIR} ${PROCESSED_DIR}

GENOMES_PER_JOB=10

for i in $(seq 0 $((GENOMES_PER_JOB - 1))); do
    GLOBAL_ID=$((SLURM_ARRAY_TASK_ID * GENOMES_PER_JOB + i))
    GENOME_ID=$(printf "%05d" ${GLOBAL_ID})
    PREFIX=${RAW_DIR}/genome_${GENOME_ID}

    echo "=== Genome ${GENOME_ID} 시뮬레이션 시작 ==="

    python ${PROJECT}/data/simulate/simulate_exp4_gc_r.py \
        --tree_path  ${TREE} \
        --out_prefix ${PREFIX} \
        --seed       ${GLOBAL_ID}

    python ${PROJECT}/data/simulate/fasta_to_npz.py \
        --fasta ${PREFIX}.fasta \
        --pi    ${PREFIX}_pi.txt \
        --out   ${PROCESSED_DIR}/genome_${GENOME_ID}.npz

    echo "=== Genome ${GENOME_ID} 완료 ==="
done
