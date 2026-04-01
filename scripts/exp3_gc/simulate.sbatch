#!/bin/bash --login
# scripts/exp3_gc/simulate.sbatch
#
# [Exp3: gc] GC OU process 기반 연속 π 변화 시뮬레이션. rate variation 없음.
# GC(x) ~ OU(mu=0.41, sigma=0.07, theta=1/100000), π(x) = f(GC(x)).
# 각 파일 = 긴 genome 서열, sliding window로 훈련 시 분해.
#
# [규모]
#   --array=0-999  → 1,000 jobs × 10 genomes = 10,000 genomes
#   genome ID: job_id * 10 + i  (00000 ~ 09999)
#
# [전체 파이프라인]
#   1. sbatch --array=0-999 scripts/exp3_gc/simulate.sbatch
#   2. sbatch scripts/exp3_gc/train_f81_gc.sbatch  (+ supervised, naive)
#
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH -A liulab

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/exp3_gc/raw
PROCESSED_DIR=${PROJECT}/data/exp3_gc/processed

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

    python ${PROJECT}/data/simulate/simulate_exp3_gc.py \
        --tree_path  ${TREE} \
        --out_prefix ${PREFIX} \
        --seed       ${GLOBAL_ID}

    python ${PROJECT}/data/simulate/fasta_to_npz.py \
        --fasta ${PREFIX}.fasta \
        --pi    ${PREFIX}_pi.txt \
        --out   ${PROCESSED_DIR}/genome_${GENOME_ID}.npz

    echo "=== Genome ${GENOME_ID} 완료 ==="
done
