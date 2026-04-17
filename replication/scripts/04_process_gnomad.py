"""
scripts/04_process_gnomad.py

gnomAD v3.1.2 (hg38) per-chromosome VCF → parquet
MAF threshold별 negative set 생성 (Figure 3b용)

사용법:
    python scripts/04_process_gnomad.py \
        --chrom 1 \
        --out_dir data/processed/gnomad \
        --min_an 25000

    # 전체 chromosome 처리 (HPCC array job 권장):
    for i in {1..22} X; do
        python scripts/04_process_gnomad.py --chrom $i --out_dir data/processed/gnomad
    done

gnomAD VCF 다운로드 경로:
    gs://gcp-public-data--gnomad/release/3.1.2/vcf/genomes/
    파일명: gnomad.genomes.v3.1.2.sites.chr{CHROM}.vcf.bgz (~5-20GB/chrom)
"""
import argparse
import gzip
import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm

GNOMAD_GCS_BASE = (
    "gs://gcp-public-data--gnomad/release/3.1.2/vcf/genomes/"
    "gnomad.genomes.v3.1.2.sites.chr{chrom}.vcf.bgz"
)

MAF_THRESHOLDS = [0.001, 0.005, 0.01, 0.05]


def download_gnomad_chr(chrom: str, out_dir: str) -> str:
    """gsutil로 gnomAD VCF 다운로드 (없으면). 로컬 경로 반환."""
    vcf_path = os.path.join(out_dir, f"gnomad.chr{chrom}.vcf.bgz")
    if not os.path.exists(vcf_path):
        gcs_url = GNOMAD_GCS_BASE.format(chrom=chrom)
        print(f"Downloading gnomAD chr{chrom} from GCS...")
        subprocess.run(["gsutil", "-m", "cp", gcs_url, vcf_path], check=True)
    return vcf_path


def parse_gnomad_chr(vcf_path: str, chrom: str, min_an: int = 25000) -> pd.DataFrame:
    """gnomAD VCF 파싱 → DataFrame (SNV, PASS, AN≥min_an)."""
    rows = []
    opener = gzip.open if vcf_path.endswith(".gz") or vcf_path.endswith(".bgz") else open

    with opener(vcf_path, "rt") as f:
        for line in tqdm(f, desc=f"Parsing chr{chrom}", unit=" lines", mininterval=5.0):
            if line.startswith("#"):
                continue

            cols = line.rstrip().split("\t", 8)
            if len(cols) < 8:
                continue

            chrom_col, pos_str, _, ref, alt_str, _, flt, info_str = cols[:8]

            # PASS only
            if flt != "PASS":
                continue

            # 단일 SNV
            if "," in alt_str:
                continue
            alt = alt_str
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                continue

            # INFO 파싱 (필요한 필드만)
            af = an = ac = None
            for field in info_str.split(";"):
                if field.startswith("AF="):
                    try:
                        af = float(field[3:].split(",")[0])
                    except ValueError:
                        pass
                elif field.startswith("AN="):
                    try:
                        an = int(field[3:])
                    except ValueError:
                        pass
                elif field.startswith("AC="):
                    try:
                        ac = int(field[3:].split(",")[0])
                    except ValueError:
                        pass

            if af is None or an is None:
                continue

            # AN 필터 (충분한 allele count 보유)
            if an < min_an:
                continue

            # MAF = min(AF, 1-AF)
            maf = min(af, 1 - af)

            rows.append({
                "chrom": chrom,
                "pos":   int(pos_str),
                "ref":   ref,
                "alt":   alt,
                "af":    af,
                "maf":   maf,
                "an":    an,
                "ac":    ac if ac is not None else -1,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Process gnomAD VCF → parquet")
    parser.add_argument("--chrom", required=True, help="염색체 (e.g. 1, 22, X)")
    parser.add_argument("--out_dir", required=True, help="출력 디렉토리")
    parser.add_argument("--vcf", default=None,
                        help="로컬 VCF 경로 (없으면 GCS에서 다운로드)")
    parser.add_argument("--min_an", type=int, default=25000,
                        help="최소 allele number (default: 25000)")
    parser.add_argument("--skip_download", action="store_true",
                        help="GCS 다운로드 건너뜀 (로컬 VCF 경로 필요)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(os.path.dirname(args.out_dir),
                           "..", "raw", "gnomad")
    os.makedirs(raw_dir, exist_ok=True)

    # VCF 경로 확보
    if args.vcf:
        vcf_path = args.vcf
    elif not args.skip_download:
        vcf_path = download_gnomad_chr(args.chrom, raw_dir)
    else:
        print("--vcf 또는 --skip_download 없이는 VCF를 찾을 수 없습니다.")
        sys.exit(1)

    # 파싱
    df = parse_gnomad_chr(vcf_path, args.chrom, min_an=args.min_an)
    print(f"\nchr{args.chrom} 총 SNV: {len(df):,}")

    # MAF threshold별 subset 저장
    for thr in MAF_THRESHOLDS:
        subset = df[df.maf >= thr].copy()
        out_path = os.path.join(args.out_dir, f"gnomad_chr{args.chrom}_maf{thr}.parquet")
        subset.to_parquet(out_path, index=False)
        print(f"  MAF≥{thr}: {len(subset):,} → {out_path}")

    # 전체도 저장
    out_all = os.path.join(args.out_dir, f"gnomad_chr{args.chrom}_all.parquet")
    df.to_parquet(out_all, index=False)
    print(f"\n전체 저장: {out_all}")


if __name__ == "__main__":
    main()
