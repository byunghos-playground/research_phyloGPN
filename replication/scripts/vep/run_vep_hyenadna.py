"""
scripts/vep/run_vep_hyenadna.py

HyenaDNA (LongSafari/hyenadna-medium-160k-seqlen-hf) variant effect prediction.

HyenaDNA는 인과적(causal) 언어 모델이므로, 변이 위치의 조건부 log-likelihood로 LLR 계산:
  LLR = log P(alt | left_context) - log P(ref | left_context)

  즉, 변이 위치 직전까지의 시퀀스로 조건부 확률 계산.
  입력: [pos-1000 ~ pos-1] → 모델의 마지막 토큰 예측 logits에서
        P(alt) - P(ref) 계산.

사용법:
    python scripts/vep/run_vep_hyenadna.py \
        --variants data/processed/clinvar.parquet \
        --genome   data/raw/hg38/hg38.fa \
        --out      results/preds/clinvar/hyenadna.parquet \
        --batch_size 64 \
        --device cuda
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LongSafari/hyenadna-medium-160k-seqlen-hf"
LEFT_CONTEXT = 999  # 변이 위치 직전까지 1000bp (변이 포함 X)


def load_model(device: str):
    print(f"Loading HyenaDNA from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    return tokenizer, model


def extract_left_context(
    genome: Fasta, chrom: str, pos: int, context_len: int
) -> str | None:
    """변이 위치(pos, 1-indexed) 직전 context_len bp 추출."""
    end   = pos - 1          # 0-indexed exclusive (= 변이 위치 바로 앞)
    start = end - context_len

    chr_key = f"chr{chrom}" if f"chr{chrom}" in genome else chrom
    if chr_key not in genome:
        return None
    if start < 0:
        return None

    return str(genome[chr_key][start:end]).upper()


def compute_llr_batch(
    left_contexts: list[str],
    refs: list[str],
    alts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    """마지막 토큰 위치의 조건부 log P(alt) - log P(ref)."""
    enc = tokenizer(
        left_contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=LEFT_CONTEXT,
    )
    input_ids = enc["input_ids"].to(device)
    # HyenaDNA tokenizer는 attention_mask를 반환하지 않을 수 있음
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if attention_mask is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids)
    logits = outputs.logits  # causal LM: logits[i, t] = P(t+1 | 0..t)

    results = []
    for i, (ref, alt) in enumerate(zip(refs, alts)):
        if attention_mask is not None:
            seq_len = attention_mask[i].sum().item()
        else:
            seq_len = input_ids.shape[1]
        last_pos = seq_len - 1

        ref_id = tokenizer.convert_tokens_to_ids(ref.upper())
        alt_id = tokenizer.convert_tokens_to_ids(alt.upper())

        if ref_id is None or alt_id is None:
            results.append(None)
            continue

        log_probs = F.log_softmax(logits[i, last_pos], dim=-1)
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

    llrs = []
    batch = {"contexts": [], "refs": [], "alts": [], "idx": []}

    def flush():
        if not batch["contexts"]:
            return
        scores = compute_llr_batch(
            batch["contexts"], batch["refs"], batch["alts"],
            tokenizer, model, device,
        )
        for orig_i, s in zip(batch["idx"], scores):
            llrs.append((orig_i, s))
        for k in batch:
            batch[k].clear()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="HyenaDNA VEP"):
        ctx = extract_left_context(genome, str(row.chrom), int(row.pos), LEFT_CONTEXT)
        if ctx is None:
            llrs.append((i, None))
            continue

        batch["contexts"].append(ctx)
        batch["refs"].append(row.ref)
        batch["alts"].append(row.alt)
        batch["idx"].append(i)

        if len(batch["contexts"]) >= args.batch_size:
            flush()

    flush()

    llr_map = {idx: score for idx, score in llrs}
    df["llr_hyenadna"] = [llr_map.get(i) for i in df.index]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    valid = df["llr_hyenadna"].notna().sum()
    print(f"\n완료: {valid:,}/{len(df):,}")
    print(f"저장: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",   required=True)
    parser.add_argument("--genome",     required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run_vep(args)


if __name__ == "__main__":
    main()
