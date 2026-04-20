#!/bin/bash
export HF_HOME=/mnt/research/liulab/RA_lee/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/research/liulab/RA_lee/.cache/huggingface
PY=/mnt/research/liulab/RA_lee/.conda/envs/phylogpn-replication/bin/python3
PROJECT=/mnt/research/liulab/RA_lee/projects/research_phyloGPN

cd ${PROJECT}

$PY -c "import pandas as pd; pd.read_parquet('replication/data/processed/clinvar.parquet').head(100).to_parquet('/tmp/test_100.parquet', index=False)"

$PY replication/scripts/vep/run_vep_hyenadna.py \
    --variants /tmp/test_100.parquet \
    --genome replication/data/raw/hg38/hg38.fa \
    --out /tmp/test_hyena.parquet \
    --batch_size 8 --device cuda

$PY -c "
import pandas as pd
r = pd.read_parquet('/tmp/test_hyena.parquet')
print(r['llr_hyenadna'].notna().sum(), '/ 100 scored')
print(r['llr_hyenadna'].describe())
"
