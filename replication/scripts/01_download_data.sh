#!/bin/bash
# scripts/01_download_data.sh
#
# 필요한 데이터를 다운로드한다.
#
# 실행:
#   bash scripts/01_download_data.sh
#
# 용량 안내:
#   hg38.fa.gz      ~900MB
#   clinvar VCF     ~50MB
#   gnomAD (per chr) ~5-20GB × 24 → 로컬보다 HPCC에서 직접 실행 권장
#   OMIM Excel      ~1MB (수동 다운로드 필요 - 아래 주석 참조)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$REPO_ROOT/data/raw"

# ─────────────────────────────────────────────
# 1. hg38 reference genome (UCSC)
# ─────────────────────────────────────────────
HG38_DIR="$RAW/hg38"
if [ ! -f "$HG38_DIR/hg38.fa.gz" ]; then
    echo "[1/3] Downloading hg38..."
    wget -P "$HG38_DIR" \
        https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    wget -P "$HG38_DIR" \
        https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz.fai 2>/dev/null || true
else
    echo "[1/3] hg38 already exists, skipping."
fi

# pyfaidx로 인덱스 생성 (samtools 불필요)
if [ ! -f "$HG38_DIR/hg38.fa.fai" ]; then
    echo "  Decompressing hg38..."
    gunzip -c "$HG38_DIR/hg38.fa.gz" > "$HG38_DIR/hg38.fa"
    echo "  Indexing hg38 with pyfaidx..."
    python -c "from pyfaidx import Fasta; Fasta('$HG38_DIR/hg38.fa'); print('  Index done.')"
fi

# ─────────────────────────────────────────────
# 2. ClinVar VCF (GRCh38, 2023-07-30 release)
# ─────────────────────────────────────────────
CLINVAR_DIR="$RAW/clinvar"
CLINVAR_RELEASE="20230730"
CLINVAR_VCF="clinvar_${CLINVAR_RELEASE}.vcf.gz"

if [ ! -f "$CLINVAR_DIR/$CLINVAR_VCF" ]; then
    echo "[2/3] Downloading ClinVar VCF..."
    wget -P "$CLINVAR_DIR" \
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/${CLINVAR_VCF}"
    wget -P "$CLINVAR_DIR" \
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/${CLINVAR_VCF}.tbi"
else
    echo "[2/3] ClinVar VCF already exists, skipping."
fi

# ─────────────────────────────────────────────
# 3. OMIM regulatory variants (Smedley et al. 2016)
# ─────────────────────────────────────────────
# NOTE: 수동 다운로드 필요.
# Smedley et al. 2016 PLOS Genetics, Table S6 (Excel):
#   https://doi.org/10.1371/journal.pgen.1006176
#   → Supplementary Material → S6 Table
# 다운로드 후 아래 경로에 저장:
#   data/raw/omim/smedley2016_tableS6.xlsx
OMIM_DIR="$RAW/omim"
if [ ! -f "$OMIM_DIR/smedley2016_tableS6.xlsx" ]; then
    echo "[3/3] OMIM: 수동 다운로드 필요."
    echo "  → https://doi.org/10.1371/journal.pgen.1006176 에서"
    echo "    Table S6 (Excel)을 다운로드하여"
    echo "    data/raw/omim/smedley2016_tableS6.xlsx 에 저장하세요."
else
    echo "[3/3] OMIM Excel already exists."
fi

# ─────────────────────────────────────────────
# 4. gnomAD (per-chromosome, GCS)
# ─────────────────────────────────────────────
# 용량이 크므로 HPCC에서 04_process_gnomad.py로 직접 처리 권장.
# 여기서는 다운로드 명령어만 출력.
echo ""
echo "gnomAD 다운로드는 04_process_gnomad.py가 처리합니다."
echo "  (각 chr VCF ~5-20GB, HPCC 실행 권장)"

echo ""
echo "=== 다운로드 완료 ==="
