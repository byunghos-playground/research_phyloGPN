"""
scripts/02_process_clinvar.py

ClinVar VCF → parquet

출력 컬럼: chrom, pos, ref, alt, label (Pathogenic/Benign), clnsig, mc

사용법:
    python scripts/02_process_clinvar.py \
        --vcf  data/raw/clinvar/clinvar_20230730.vcf.gz \
        --out  data/processed/clinvar.parquet
"""
import argparse
import gzip
import sys
import pandas as pd
from tqdm import tqdm

CHROMS = {str(i) for i in range(1, 23)} | {"X"}

# ClinVar CLNREVSTAT → star 등급 매핑
# at least 1 star: criteria_provided_* 이상
ZERO_STAR = {
    "no_assertion_criteria_provided",
    "no_classification_provided",
    "no_assertion_provided",
    "no_classifications_from_unflagged_records",
}


def parse_info(info_str):
    d = {}
    for field in info_str.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            d[k] = v
    return d


def assign_label(clnsig: str, include_likely: bool) -> str | None:
    """Pathogenic/Benign 레이블 반환. 해당 없으면 None."""
    sig = clnsig.replace("_", " ").replace("/", "|")
    is_path = "Pathogenic" in sig
    is_likely_path = "Likely pathogenic" in sig
    is_benign = "Benign" in sig
    is_likely_benign = "Likely benign" in sig

    if include_likely:
        pos = is_path or is_likely_path
        neg = is_benign or is_likely_benign
    else:
        pos = is_path and not is_likely_path
        neg = is_benign and not is_likely_benign

    # 충돌하면 제외
    if pos and neg:
        return None
    if pos:
        return "Pathogenic"
    if neg:
        return "Benign"
    return None


def parse_clinvar(vcf_path: str, include_likely: bool = True) -> pd.DataFrame:
    rows = []
    opener = gzip.open if vcf_path.endswith(".gz") else open

    with opener(vcf_path, "rt") as f:
        for line in tqdm(f, desc="Parsing ClinVar VCF", unit=" lines"):
            if line.startswith("#"):
                continue

            cols = line.rstrip().split("\t")
            if len(cols) < 8:
                continue

            chrom, pos_str, _, ref, alt_str = cols[:5]
            info_str = cols[7]

            # chrom 정규화
            chrom = chrom.replace("chr", "")
            if chrom not in CHROMS:
                continue

            # 단일 대립유전자
            if "," in alt_str:
                continue
            alt = alt_str

            # SNV: ref/alt 각 1bp, N 제외
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref == "N" or alt == "N":
                continue

            info = parse_info(info_str)

            # single_nucleotide_variant만
            if info.get("CLNVC", "") != "single_nucleotide_variant":
                continue

            # CLNREVSTAT 기반 최소 1-star 필터
            revstat = info.get("CLNREVSTAT", "")
            if revstat in ZERO_STAR:
                continue

            # 레이블 결정
            clnsig = info.get("CLNSIG", "")
            label = assign_label(clnsig, include_likely)
            if label is None:
                continue

            # Molecular consequence (마지막 파트 추출)
            mc_raw = info.get("MC", "")
            mc = mc_raw.split("|")[-1] if "|" in mc_raw else mc_raw

            rows.append({
                "chrom": chrom,
                "pos": int(pos_str),
                "ref": ref,
                "alt": alt,
                "label": label,
                "clnsig": clnsig,
                "mc": mc,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Process ClinVar VCF → parquet")
    parser.add_argument("--vcf", required=True, help="ClinVar VCF (.vcf.gz)")
    parser.add_argument("--out", required=True, help="출력 parquet 경로")
    parser.add_argument(
        "--include_likely",
        action="store_true",
        default=True,
        help="Likely_pathogenic / Likely_benign 포함 (default: True)",
    )
    args = parser.parse_args()

    df = parse_clinvar(args.vcf, include_likely=args.include_likely)

    print(f"\n총 변이: {len(df):,}")
    print(f"  Pathogenic : {(df.label == 'Pathogenic').sum():,}")
    print(f"  Benign     : {(df.label == 'Benign').sum():,}")
    print(f"\nMolecular consequence top-10:")
    print(df.mc.value_counts().head(10).to_string())

    df.to_parquet(args.out, index=False)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
