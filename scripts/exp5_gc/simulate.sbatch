#!/bin/bash --login
# scripts/exp5_gc/simulate.sbatch
# [Exp5: gc, branch_scale=10] GC OU + F81, branch lengths ×10
#
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-99
#SBATCH -A liulab
#SBATCH --output=logs/exp5_gc/simulate_%A_%a.out
#SBATCH --error=logs/exp5_gc/simulate_%A_%a.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/exp5_gc/raw
PROCESSED_DIR=${PROJECT}/data/exp5_gc/processed

module purge
module load Miniforge3
conda activate phylogpn-env

mkdir -p ${RAW_DIR} ${PROCESSED_DIR}

GENOMES_PER_JOB=30

for i in $(seq 0 $((GENOMES_PER_JOB - 1))); do
    GLOBAL_ID=$((SLURM_ARRAY_TASK_ID * GENOMES_PER_JOB + i))
    GENOME_ID=$(printf "%05d" ${GLOBAL_ID})
    PREFIX=${RAW_DIR}/genome_${GENOME_ID}

    python ${PROJECT}/data/simulate/simulate_exp3_gc.py \
        --tree_path    ${TREE} \
        --L            10000 \
        --out_prefix   ${PREFIX} \
        --seed         ${GLOBAL_ID} \
        --branch_scale 10.0

    python ${PROJECT}/data/simulate/fasta_to_npz.py \
        --fasta  ${PREFIX}.fasta \
        --pi     ${PREFIX}_pi.txt \
        --out    ${PROCESSED_DIR}/genome_${GENOME_ID}.npz
done

echo "=== [Exp5] simulate array ${SLURM_ARRAY_TASK_ID} 완료 ==="
