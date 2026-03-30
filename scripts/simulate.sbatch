#!/bin/bash --login
# scripts/simulate.sbatch
#
# F81 시뮬레이션 SLURM array job.
# 각 task = 하나의 chunk (L=481 사이트, 청크당 π 하나).
# array job은 병렬로 실행되므로 청크 수에 관계없이 벽시계 시간은 동일(~1시간).
#
# [권장 규모]
#   --array=0-9999  → 10,000 chunks × 481 사이트
#                     train 8,000 / valid 1,000 / test 1,000 청크
#
# [전체 파이프라인]
#   1. sbatch --array=0-9999 scripts/simulate.sbatch  # 시뮬레이션 + npz 변환
#   2. (array 완료 후) python data/simulate/split_data.py \
#          --processed_dir data/processed \
#          --copy          # HPCC NFS: symlink 대신 복사 권장
#   3. sbatch scripts/train_f81.sbatch                # 훈련
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

CHUNK_ID=$(printf "%05d" ${SLURM_ARRAY_TASK_ID})
PREFIX=${RAW_DIR}/chunk_${CHUNK_ID}

echo "=== Chunk ${CHUNK_ID} 시뮬레이션 시작 ==="

# 1) F81 시뮬레이션
python ${PROJECT}/data/simulate/simulate_f81.py \
    --tree_path ${TREE} \
    --L         481 \
    --out_prefix ${PREFIX} \
    --seed      ${SLURM_ARRAY_TASK_ID}

# 2) FASTA + pi → .npz 변환
python ${PROJECT}/data/simulate/fasta_to_npz.py \
    --fasta ${PREFIX}.fasta \
    --pi    ${PREFIX}_pi.txt \
    --out   ${PROCESSED_DIR}/block_${CHUNK_ID}.npz

echo "=== Chunk ${CHUNK_ID} 완료 ==="
