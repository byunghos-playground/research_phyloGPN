#!/bin/bash --login
# scripts/simulate.sbatch
#
# F81 시뮬레이션 SLURM array job.
# 각 task = 하나의 chunk (L=10000 사이트).
# array 범위를 조정하면 총 시뮬레이션 사이트 수 변경 가능.
#   예: --array=0-99 → 100 chunks × 10000 = 1,000,000 사이트
#
# 제출 방법:
#   sbatch --array=0-9 scripts/simulate.sbatch
#
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH -A liulab

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/raw
PROCESSED_DIR=${PROJECT}/data/processed

module purge
module load Miniforge3
conda activate phylogpn-env

mkdir -p ${RAW_DIR} ${PROCESSED_DIR}

CHUNK_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
PREFIX=${RAW_DIR}/chunk_${CHUNK_ID}

echo "=== Chunk ${CHUNK_ID} 시뮬레이션 시작 ==="

# 1) F81 시뮬레이션
python ${PROJECT}/data/simulate/simulate_f81.py \
    --tree_path ${TREE} \
    --L         10000 \
    --out_prefix ${PREFIX} \
    --seed      ${SLURM_ARRAY_TASK_ID}

# 2) FASTA + pi → .npz 변환
python ${PROJECT}/data/simulate/fasta_to_npz.py \
    --fasta ${PREFIX}.fasta \
    --pi    ${PREFIX}_pi.txt \
    --out   ${PROCESSED_DIR}/block_${CHUNK_ID}.npz

echo "=== Chunk ${CHUNK_ID} 완료 ==="
