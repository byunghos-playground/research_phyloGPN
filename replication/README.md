# PhyloGPN Paper Replication — Figure 3

Albors et al. 2025 (RECOMB), *A Phylogenetic Approach to Genomic Language Modeling*  
[PMC11908359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11908359/)

---

## 목표

- **Figure 3a**: ClinVar Pathogenic vs Benign ROC curves (5 모델)
- **Figure 3b**: OMIM regulatory variants vs gnomAD AUPRC at multiple MAF thresholds

## 비교 모델

| Key | HuggingFace ID | LLR 방식 |
|-----|---------------|---------|
| PhyloGPN | `songlab/PhyloGPN` | F81 logit[alt] − logit[ref] |
| GPN-MSA | `songlab/gpn-msa-sapiens` | Masked LM log P(alt) − log P(ref) |
| NT | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | Masked LM (6-mer token) |
| HyenaDNA | `LongSafari/hyenadna-large-1m-seqlen-hf` | Causal LM log P(alt\|ctx) − log P(ref\|ctx) |
| Caduceus | `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` | Masked LM + RC 평균 |

---

## 실행 순서

### 0. 환경 설정
```bash
conda env create -f replication/envs/environment.yaml
conda activate phylogpn-replication
```

### 1. 데이터 다운로드
```bash
bash replication/scripts/01_download_data.sh
```
> OMIM Excel (`smedley2016_tableS6.xlsx`) 은 수동 다운로드 필요:  
> https://doi.org/10.1371/journal.pgen.1006176 → S6 Table → `data/raw/omim/`

### 2. 데이터 전처리
```bash
# ClinVar
python replication/scripts/02_process_clinvar.py \
    --vcf replication/data/raw/clinvar/clinvar_20230730.vcf.gz \
    --out replication/data/processed/clinvar.parquet

# OMIM
python replication/scripts/03_process_omim.py \
    --xlsx replication/data/raw/omim/smedley2016_tableS6.xlsx \
    --out  replication/data/processed/omim.parquet

# gnomAD (HPCC 권장, 각 chr 5-20GB)
for i in {1..22} X; do
    python replication/scripts/04_process_gnomad.py \
        --chrom $i \
        --out_dir replication/data/processed/gnomad
done
```

### 3. VEP 추론 (HPCC)
```bash
# 모두 독립 실행 가능 (병렬 제출 권장)
sbatch replication/scripts/sbatch/vep_phylogpn.sbatch
sbatch replication/scripts/sbatch/vep_gpn_msa.sbatch
sbatch replication/scripts/sbatch/vep_nt.sbatch
sbatch replication/scripts/sbatch/vep_hyenadna.sbatch
sbatch replication/scripts/sbatch/vep_caduceus.sbatch
```

### 4. Figure 3 플롯
```
replication/notebooks/figure3.ipynb
```
출력: `replication/results/figures/figure3_combined.pdf`

---

## 디렉토리 구조

```
replication/
├── envs/
│   └── environment.yaml
├── data/
│   ├── raw/          # 원본 데이터 (git 미추적)
│   └── processed/    # parquet (git 미추적)
├── results/
│   └── preds/        # 모델별 LLR 예측 (git 미추적)
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_process_clinvar.py
│   ├── 03_process_omim.py
│   ├── 04_process_gnomad.py
│   ├── vep/          # 모델별 VEP 추론 스크립트
│   └── sbatch/       # HPCC 제출 스크립트
└── notebooks/
    └── figure3.ipynb
```

## 주의사항

- **GPN-MSA**: `songlab/gpn-msa-sapiens` 는 MSA 입력 기반 모델.  
  `run_vep_gpn_msa.py` 는 sequence-only fallback 사용 → 논문 결과와 차이 가능.  
  정확한 재현을 위해서는 `gpn.msa.inference vep` CLI + multiz100way MSA 필요.

- **gnomAD**: 전체 다운로드 ~200GB+ → HPCC scratch 디렉토리 권장.

- **OMIM**: 논문에서 사용한 409 변이는 Smedley et al. 2016 Table S6 기반.  
  liftover 후 SNV 필터링 결과 수가 정확히 409가 아닐 수 있음 (liftover 실패 제외).
