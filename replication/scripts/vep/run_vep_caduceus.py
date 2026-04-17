"""
scripts/vep/run_vep_caduceus.py

Caduceus (kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16)
variant effect prediction.

Caduceus는 Masked LM + Reverse Complement (RC) 대칭성:
  - 변이 위치를 MASK로 교체
  - Forward sequence AND reverse complement sequence 각각 실행
  - LLR = 0.5 * [(log P_fwd(alt) - log P_fwd(ref))
                + (log P_rc(rc_alt) - log P_rc(rc_ref))]

사용법:
    python scripts/vep/run_vep_caduceus.py \
        --variants data/processed/clinvar.parquet \
        --genome   data/raw/hg38/hg38.fa \
        --out      results/preds/clinvar/caduceus.parquet \
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

MODEL_ID = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
WINDOW = 512

RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def rc(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1]


def load_model(device: str):
    print(f"Loading Caduceus from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    return tokenizer, model


def extract_seq(genome: Fasta, chrom: str, pos: int) -> tuple[str, int] | tuple[None, None]:
    half = WINDOW // 2
    start = pos - 1 - half
    end   = pos - 1 + WINDOW - half

    chr_key = f"chr{chrom}" if f"chr{chrom}" in genome else chrom
    if chr_key not in genome:
        return None, None

    chr_len = len(genome[chr_key])
    if start < 0 or end > chr_len:
        return None, None

    seq = str(genome[chr_key][start:end]).upper()
    center_idx = pos - 1 - start  # 0-indexed
    return seq, center_idx


def masked_log_prob(
    seqs: list[str],
    mask_idxs: list[int],
    target_nucs: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    """mask_idxs 위치를 MASK로 교체 후 log P(target_nuc) 반환."""
    masked = []
    for seq, midx in zip(seqs, mask_idxs):
        s = list(seq)
        s[midx] = tokenizer.mask_token
        masked.append("".join(s))

    enc = tokenizer(masked, return_tensors="pt", padding=True,
                    truncation=True, max_length=WINDOW + 2)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, vocab)

    log_probs_list = []
    for i, (midx, nuc) in enumerate(zip(mask_idxs, target_nucs)):
        nuc_id = tokenizer.convert_tokens_to_ids(nuc.upper())
        if nuc_id is None:
            log_probs_list.append(None)
            continue
        # mask token 위치 (CLS offset 고려)
        cls_offset = 1 if tokenizer.cls_token_id is not None else 0
        tok_pos = midx + cls_offset
        if tok_pos >= logits.shape[1]:
            log_probs_list.append(None)
            continue
        lp = F.log_softmax(logits[i, tok_pos], dim=-1)[nuc_id].item()
        log_probs_list.append(lp)

    return log_probs_list


def compute_llr_batch(
    seqs: list[str],
    center_idxs: list[int],
    refs: list[str],
    alts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    B = len(seqs)

    # Forward pass
    fwd_ref_lp  = masked_log_prob(seqs, center_idxs, refs, tokenizer, model, device)
    fwd_alt_lp  = masked_log_prob(seqs, center_idxs, alts, tokenizer, model, device)

    # Reverse complement pass
    rc_seqs       = [rc(s) for s in seqs]
    rc_center_idxs = [len(s) - 1 - c for s, c in zip(seqs, center_idxs)]
    rc_refs       = [rc(r) for r in refs]
    rc_alts       = [rc(a) for a in alts]

    rc_ref_lp = masked_log_prob(rc_seqs, rc_center_idxs, rc_refs, tokenizer, model, device)
    rc_alt_lp = masked_log_prob(rc_seqs, rc_center_idxs, rc_alts, tokenizer, model, device)

    results = []
    for fref, falt, rref, ralt in zip(fwd_ref_lp, fwd_alt_lp, rc_ref_lp, rc_alt_lp):
        if any(x is None for x in [fref, falt, rref, ralt]):
            results.append(None)
        else:
            llr = 0.5 * ((falt - fref) + (ralt - rref))
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

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Caduceus VEP"):
        seq, cidx = extract_seq(genome, str(row.chrom), int(row.pos))
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
    df["llr_caduceus"] = [llr_map.get(i) for i in df.index]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    valid = df["llr_caduceus"].notna().sum()
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
