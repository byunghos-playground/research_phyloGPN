"""
scripts/vep/run_vep_nt.py

Nucleotide Transformer (InstaDeepAI/nucleotide-transformer-2.5b-multi-species)
variant effect prediction.

NT는 non-overlapping 6-mer 토크나이저를 사용:
  - 128 tokens = 768bp 입력
  - 변이 위치를 포함하는 6-mer 토큰을 MASK로 교체
  - LLR = log P(6-mer with alt) - log P(6-mer with ref) at masked position

사용법:
    python scripts/vep/run_vep_nt.py \
        --variants data/processed/clinvar.parquet \
        --genome   data/raw/hg38/hg38.fa \
        --out      results/preds/clinvar/nt.parquet \
        --batch_size 32 \
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

MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
K = 6              # k-mer size
N_TOKENS = 128     # number of k-mer tokens
SEQ_LEN = N_TOKENS * K  # 768 bp


def load_model(device: str):
    print(f"Loading NT from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return tokenizer, model


def extract_seq(genome: Fasta, chrom: str, pos: int) -> tuple[str, int] | tuple[None, None]:
    """768bp 윈도우 추출. center = pos (1-indexed). 반환: (seq, pos_in_seq 0-indexed)"""
    half = SEQ_LEN // 2
    start = pos - 1 - half
    end   = pos - 1 + SEQ_LEN - half

    chr_key = f"chr{chrom}" if f"chr{chrom}" in genome else chrom
    if chr_key not in genome:
        return None, None

    chr_len = len(genome[chr_key])
    if start < 0 or end > chr_len:
        return None, None

    seq = str(genome[chr_key][start:end]).upper()
    pos_in_seq = pos - 1 - start  # 0-indexed position of variant in seq
    return seq, pos_in_seq


def make_kmer_token(seq: str, tok_idx: int) -> str:
    """tok_idx번째 6-mer 문자열 반환."""
    return seq[tok_idx * K: (tok_idx + 1) * K]


def compute_llr_batch(
    seqs: list[str],
    pos_in_seqs: list[int],
    refs: list[str],
    alts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    results = []

    # 변이 위치를 포함하는 6-mer 토큰 인덱스 계산
    tok_idxs = [p // K for p in pos_in_seqs]  # 0-indexed token position (no CLS)

    # MASK된 시퀀스 생성 (해당 6-mer → NNNNNN or mask token string)
    masked_seqs = []
    for seq, tok_idx in zip(seqs, tok_idxs):
        seq_list = list(seq)
        for j in range(K):
            seq_list[tok_idx * K + j] = "N"  # N으로 마스킹 → tokenizer가 [MASK]로 인식
        masked_seqs.append("".join(seq_list))

    # tokenize
    enc = tokenizer(
        masked_seqs,
        return_tensors="pt",
        padding="max_length",
        max_length=N_TOKENS + 2,  # +2 for special tokens if any
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L_tokens, vocab)

    for i, (seq, pos_in_seq, ref, alt, tok_idx) in enumerate(
        zip(seqs, pos_in_seqs, refs, alts, tok_idxs)
    ):
        # ref kmer vs alt kmer
        ref_kmer = make_kmer_token(seq, tok_idx)
        alt_seq = list(seq)
        alt_seq[pos_in_seq] = alt.upper()
        alt_kmer = make_kmer_token("".join(alt_seq), tok_idx)

        # token IDs for kmers
        ref_kmer_id = tokenizer.convert_tokens_to_ids(ref_kmer)
        alt_kmer_id = tokenizer.convert_tokens_to_ids(alt_kmer)

        if ref_kmer_id is None or alt_kmer_id is None:
            results.append(None)
            continue

        # token position in model output (may have CLS offset)
        # NT typically has no CLS; check tokenizer
        cls_offset = 1 if tokenizer.cls_token else 0
        model_tok_pos = tok_idx + cls_offset

        if model_tok_pos >= logits.shape[1]:
            results.append(None)
            continue

        log_probs = F.log_softmax(logits[i, model_tok_pos], dim=-1)
        llr = (log_probs[alt_kmer_id] - log_probs[ref_kmer_id]).item()
        results.append(llr)

    return results


def run_vep(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer, model = load_model(device)
    genome = Fasta(args.genome)

    df = pd.read_parquet(args.variants)
    print(f"Variants: {len(df):,}")

    llrs = []
    batch = {"seqs": [], "pos_in_seqs": [], "refs": [], "alts": [], "idx": []}

    def flush():
        if not batch["seqs"]:
            return
        scores = compute_llr_batch(
            batch["seqs"], batch["pos_in_seqs"], batch["refs"], batch["alts"],
            tokenizer, model, device,
        )
        for orig_i, s in zip(batch["idx"], scores):
            llrs.append((orig_i, s))
        for k in batch:
            batch[k].clear()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="NT VEP"):
        seq, pos_in_seq = extract_seq(genome, str(row.chrom), int(row.pos))
        if seq is None:
            llrs.append((i, None))
            continue

        batch["seqs"].append(seq)
        batch["pos_in_seqs"].append(pos_in_seq)
        batch["refs"].append(row.ref)
        batch["alts"].append(row.alt)
        batch["idx"].append(i)

        if len(batch["seqs"]) >= args.batch_size:
            flush()

    flush()

    llr_map = {idx: score for idx, score in llrs}
    df["llr_nt"] = [llr_map.get(i) for i in df.index]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    valid = df["llr_nt"].notna().sum()
    print(f"\n완료: {valid:,}/{len(df):,}")
    print(f"저장: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",   required=True)
    parser.add_argument("--genome",     required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run_vep(args)


if __name__ == "__main__":
    main()
