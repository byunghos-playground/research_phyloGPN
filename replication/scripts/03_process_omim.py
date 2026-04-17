"""
scripts/03_process_omim.py

Smedley et al. 2016 Table S6 (Excel) → parquet
hg19 좌표를 hg38으로 liftover.

데이터 출처:
    Smedley et al. 2016, PLOS Genetics
    https://doi.org/10.1371/journal.pgen.1006176
    → Supplementary → S6 Table (Excel)

사용법:
    python scripts/03_process_omim.py \
        --xlsx data/raw/omim/smedley2016_tableS6.xlsx \
        --out  data/processed/omim.parquet
"""
import argparse
import sys
import pandas as pd
from pyliftover import LiftOver
from tqdm import tqdm

# Excel 내 시트별 variant consequence 레이블
# (실제 시트 이름은 파일을 열어 확인 필요; 아래는 일반적인 추정값)
SHEET_LABELS = {
    "promoter":  "promoter",
    "enhancer":  "enhancer",
    "5UTR":      "5UTR",
    "3UTR":      "3UTR",
    "ncRNA":     "ncRNA",
    "splice":    "splice",
    "synonymous":"synonymous",
}


def load_and_liftover(xlsx_path: str) -> pd.DataFrame:
    lo = LiftOver("hg19", "hg38")

    xl = pd.ExcelFile(xlsx_path)
    print(f"시트 목록: {xl.sheet_names}")

    all_rows = []
    for sheet in xl.sheet_names:
        df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet, header=0)
        # 컬럼명 정규화 (소문자 + strip)
        df_sheet.columns = [c.strip().lower() for c in df_sheet.columns]

        # 필수 컬럼 탐색: chromosome, position, ref, alt
        # (시트마다 컬럼명이 다를 수 있으므로 유연하게 처리)
        col_map = {}
        for c in df_sheet.columns:
            if "chr" in c:
                col_map.setdefault("chrom", c)
            elif "pos" in c or "start" in c:
                col_map.setdefault("pos", c)
            elif c in ("ref", "reference"):
                col_map["ref"] = c
            elif c in ("alt", "alternate", "mutant"):
                col_map["alt"] = c

        missing = [k for k in ["chrom", "pos", "ref", "alt"] if k not in col_map]
        if missing:
            print(f"  시트 '{sheet}': 컬럼 {missing} 없음 — 건너뜀")
            continue

        consequence = SHEET_LABELS.get(sheet.lower(), sheet)

        for _, row in tqdm(df_sheet.iterrows(), total=len(df_sheet),
                           desc=f"  Lifting '{sheet}'", leave=False):
            try:
                chrom_raw = str(row[col_map["chrom"]]).replace("chr", "").strip()
                pos_hg19 = int(row[col_map["pos"]])
                ref = str(row[col_map["ref"]]).strip().upper()
                alt = str(row[col_map["alt"]]).strip().upper()
            except (ValueError, KeyError):
                continue

            # SNV only
            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref not in "ACGT" or alt not in "ACGT":
                continue

            # liftover hg19 → hg38 (0-indexed input)
            result = lo.convert_coordinate(f"chr{chrom_raw}", pos_hg19 - 1)
            if not result:
                continue

            chrom_hg38 = result[0][0].replace("chr", "")
            pos_hg38 = result[0][1] + 1  # back to 1-indexed

            all_rows.append({
                "chrom":       chrom_hg38,
                "pos":         pos_hg38,
                "ref":         ref,
                "alt":         alt,
                "label":       "Pathogenic",
                "consequence": consequence,
            })

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Process OMIM Excel → parquet")
    parser.add_argument("--xlsx", required=True, help="Smedley 2016 Table S6 Excel 파일")
    parser.add_argument("--out", required=True, help="출력 parquet 경로")
    args = parser.parse_args()

    df = load_and_liftover(args.xlsx)

    print(f"\n총 변이 (liftover 성공): {len(df):,}")
    if len(df) > 0:
        print(f"Consequence 분포:\n{df.consequence.value_counts().to_string()}")

    df.to_parquet(args.out, index=False)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
