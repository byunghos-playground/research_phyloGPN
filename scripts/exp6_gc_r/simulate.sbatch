#!/bin/bash --login
# scripts/exp6_gc_r/simulate.sbatch
# [Exp6: gc_r, branch_scale=10] GC OU + F81 + rate variation, branch lengths ×10
#
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-99
#SBATCH -A liulab
#SBATCH --output=logs/exp6_gc_r/simulate_%A_%a.out
#SBATCH --error=logs/exp6_gc_r/simulate_%A_%a.err

PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN
TREE=${PROJECT}/data/trees/241-mammalian-2020v2.1.nh.txt
RAW_DIR=${PROJECT}/data/exp6_gc_r/raw
PROCESSED_DIR=${PROJECT}/data/exp6_gc_r/processed

module purge
module load Miniforge3
conda activate phylogpn-env

mkdir -p ${RAW_DIR} ${PROCESSED_DIR}

GENOMES_PER_JOB=30

for i in $(seq 0 $((GENOMES_PER_JOB - 1))); do
    GLOBAL_ID=$((SLURM_ARRAY_TASK_ID * GENOMES_PER_JOB + i))
    GENOME_ID=$(printf "%05d" ${GLOBAL_ID})
    PREFIX=${RAW_DIR}/genome_${GENOME_ID}

    python ${PROJECT}/data/simulate/simulate_exp4_gc_r.py \
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

echo "=== [Exp6] simulate array ${SLURM_ARRAY_TASK_ID} 완료 ==="
