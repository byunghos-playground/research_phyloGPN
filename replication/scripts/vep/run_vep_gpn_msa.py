"""
scripts/vep/run_vep_gpn_msa.py

GPN-MSA (songlab/gpn-msa-sapiens) variant effect prediction
Masked LM: mask variant position → LLR = log P(alt) - log P(ref)

NOTE: GPN-MSA는 MSA(다종 서열 정렬) 입력을 사용하는 게 정석이나,
      여기서는 gpn 패키지의 msa.inference 모듈을 사용합니다.
      MSA 데이터(songlab/multiz100way)는 HuggingFace datasets로 스트리밍됩니다.

사용법 (gpn 패키지 설치 필요):
    python -m gpn.msa.inference vep \
        data/processed/clinvar.parquet \
        data/raw/hg38/hg38.fa \
        512 \
        songlab/gpn-msa-sapiens \
        results/preds/clinvar/gpn_msa.parquet \
        --per_device_batch_size 256 \
        --dataloader_num_workers 4

또는 이 스크립트를 직접 실행 (sequence-only fallback):
    python scripts/vep/run_vep_gpn_msa.py \
        --variants data/processed/clinvar.parquet \
        --genome   data/raw/hg38/hg38.fa \
        --out      results/preds/clinvar/gpn_msa.parquet \
        --batch_size 128 \
        --device cuda
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "songlab/gpn-msa-sapiens"
WINDOW = 512

# GPN-MSA tokenizer: A=0,C=1,G=2,T=3, MASK token id는 tokenizer에서 확인
NUC_TOKENS = {"A", "C", "G", "T"}


def load_model(device: str):
    print(f"Loading GPN-MSA from {MODEL_ID}...")
    # gpn 패키지의 모델 클래스 사용
    import gpn.model  # noqa: F401 — registers GPNRoFormer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return tokenizer, model


def extract_seq(genome: Fasta, chrom: str, pos: int, window: int) -> tuple[str, int] | tuple[None, None]:
    """
    1-indexed pos 기준 window bp 추출.
    반환: (sequence, center_idx_in_seq)
    """
    half = window // 2
    start = pos - 1 - half
    end   = pos - 1 + half + (window % 2)

    chr_key = f"chr{chrom}" if f"chr{chrom}" in genome else chrom
    if chr_key not in genome:
        return None, None

    chr_len = len(genome[chr_key])
    if start < 0 or end > chr_len:
        return None, None

    seq = str(genome[chr_key][start:end]).upper()
    center_idx = pos - 1 - start   # seq 내 변이 위치 (0-indexed)
    return seq, center_idx


def compute_llr_batch(
    seqs: list[str],
    center_idxs: list[int],
    refs: list[str],
    alts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    """마스킹 기반 LLR 계산."""
    # 변이 위치를 MASK 토큰으로 교체한 시퀀스 생성
    masked_seqs = []
    for seq, cidx in zip(seqs, center_idxs):
        seq_list = list(seq)
        seq_list[cidx] = tokenizer.mask_token if hasattr(tokenizer, "mask_token") else "[MASK]"
        masked_seqs.append("".join(seq_list))

    enc = tokenizer(masked_seqs, return_tensors="pt", padding=True, truncation=True,
                    max_length=WINDOW + 2)  # +2 for [CLS]/[SEP] if any
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, vocab)

    results = []
    for i, (cidx, ref, alt) in enumerate(zip(center_idxs, refs, alts)):
        # tokenizer가 1-char 토큰이라면 cidx + CLS offset
        # GPN-MSA는 CLS 없이 시작하므로 offset=0 (확인 필요)
        tok_pos = cidx

        ref_id = tokenizer.convert_tokens_to_ids(ref.upper())
        alt_id = tokenizer.convert_tokens_to_ids(alt.upper())

        if ref_id is None or alt_id is None:
            results.append(None)
            continue

        log_probs = F.log_softmax(logits[i, tok_pos], dim=-1)
        llr = (log_probs[alt_id] - log_probs[ref_id]).item()
        results.append(llr)

    return results


def run_vep(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer, model = load_model(device)
    genome = Fasta(args.genome)

    df = pd.read_parquet(args.variants)
    print(f"Variants: {len(df):,}")

    llrs, bad_idx = [], []
    batch = {"seqs": [], "cidxs": [], "refs": [], "alts": [], "idx": []}

    def flush():
        if not batch["seqs"]:
            return
        scores = compute_llr_batch(
            batch["seqs"], batch["cidxs"], batch["refs"], batch["alts"],
            tokenizer, model, device,
        )
        for orig_i, s in zip(batch["idx"], scores):
            llrs.append((orig_i, s))
        for k in batch:
            batch[k].clear()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="GPN-MSA VEP"):
        seq, cidx = extract_seq(genome, str(row.chrom), int(row.pos), WINDOW)
        if seq is None:
            llrs.append((i, None))
            continue

        batch["seqs"].append(seq)
        batch["cidxs"].append(cidx)
        batch["refs"].append(row.ref)
        batch["alts"].append(row.alt)
        batch["idx"].append(i)

        if len(batch["seqs"]) >= args.batch_size:
            flush()

    flush()

    llr_map = {idx: score for idx, score in llrs}
    df["llr_gpn_msa"] = [llr_map.get(i) for i in df.index]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    valid = df["llr_gpn_msa"].notna().sum()
    print(f"\n완료: {valid:,}/{len(df):,} 변이 스코어링됨")
    print(f"저장: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",   required=True)
    parser.add_argument("--genome",     required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run_vep(args)


if __name__ == "__main__":
    main()
