#!/bin/bash --login
# scripts/simulate.sbatch
#
# F81 시뮬레이션 SLURM array job.
# 각 task = 30개 chunk 루프 생성 (L=481 사이트, 청크당 π 하나).
#
# [규모]
#   --array=0-999  → 1,000 jobs × 30 chunks = 30,000 chunks
#   split: train 20,000 / valid 5,000 / test 5,000
#   chunk ID: job_id * 30 + i  (00000 ~ 29999)
#
# [전체 파이프라인]
#   1. sbatch --array=0-999 scripts/simulate.sbatch
#   2. (완료 후) python data/simulate/split_data.py \
#          --processed_dir data/processed \
#          --train_ratio 0.667 --valid_ratio 0.167 \
#          --copy
#   3. sbatch scripts/train_f81.sbatch
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

CHUNKS_PER_JOB=30

for i in $(seq 0 $((CHUNKS_PER_JOB - 1))); do
    GLOBAL_ID=$((SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB + i))
    CHUNK_ID=$(printf "%05d" ${GLOBAL_ID})
    PREFIX=${RAW_DIR}/chunk_${CHUNK_ID}

    echo "=== Chunk ${CHUNK_ID} 시뮬레이션 시작 ==="

    python ${PROJECT}/data/simulate/simulate_f81.py \
        --tree_path  ${TREE} \
        --L          481 \
        --out_prefix ${PREFIX} \
        --seed       ${GLOBAL_ID}

    python ${PROJECT}/data/simulate/fasta_to_npz.py \
        --fasta ${PREFIX}.fasta \
        --pi    ${PREFIX}_pi.txt \
        --out   ${PROCESSED_DIR}/block_${CHUNK_ID}.npz

    echo "=== Chunk ${CHUNK_ID} 완료 ==="
done
