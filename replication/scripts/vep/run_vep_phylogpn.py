"""
scripts/vep/run_vep_phylogpn.py

PhyloGPN (songlab/PhyloGPN) variant effect prediction
LLR = logit[alt][center] - logit[ref][center]  (F81 rate logits 차이)

사용법:
    python scripts/vep/run_vep_phylogpn.py \
        --variants data/processed/clinvar.parquet \
        --genome   data/raw/hg38/hg38.fa \
        --out      results/preds/clinvar/phylogpn.parquet \
        --batch_size 64 \
        --device cuda
"""
import argparse
import os
import pandas as pd
import torch
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "songlab/PhyloGPN"
WINDOW = 481          # model's receptive field
PAD_HALF = WINDOW // 2   # 240

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def load_model(device: str):
    print(f"Loading PhyloGPN from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    return tokenizer, model


def extract_seq(genome: Fasta, chrom: str, pos: int, window: int) -> str | None:
    """1-indexed pos 기준으로 window bp 추출 (중심 = pos)."""
    half = window // 2
    start = pos - 1 - half   # 0-indexed
    end   = pos - 1 + half + 1

    chr_key = f"chr{chrom}" if f"chr{chrom}" in genome else chrom
    if chr_key not in genome:
        return None

    chr_len = len(genome[chr_key])
    if start < 0 or end > chr_len:
        return None

    return str(genome[chr_key][start:end]).upper()


def compute_llr_batch(
    seqs: list[str],
    refs: list[str],
    alts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[float | None]:
    """배치 내 모든 변이의 LLR을 계산."""
    pad_tok = tokenizer.pad_token
    padded = [pad_tok * PAD_HALF + s + pad_tok * PAD_HALF for s in seqs]

    enc = tokenizer(padded, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        logit_dict = model(input_ids)   # {'A': Tensor[batch, L], 'C': ..., 'G': ..., 'T': ...}

    results = []
    center = PAD_HALF + PAD_HALF  # padded sequence에서 center: 240 + 240 = 480

    for i in range(len(seqs)):
        ref = refs[i].upper()
        alt = alts[i].upper()
        if ref not in logit_dict or alt not in logit_dict:
            results.append(None)
            continue
        llr = (logit_dict[alt][i, center] - logit_dict[ref][i, center]).item()
        results.append(llr)

    return results


CHECKPOINT_INTERVAL = 100_000


def run_vep(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer, model = load_model(device)
    genome = Fasta(args.genome)

    df = pd.read_parquet(args.variants)
    print(f"Variants: {len(df):,}")

    ckpt_path = args.out + ".ckpt.parquet"
    llr_map = {}
    if os.path.exists(ckpt_path):
        ckpt = pd.read_parquet(ckpt_path)
        llr_map = dict(zip(ckpt["idx"], ckpt["llr"]))
        print(f"체크포인트 로드: {len(llr_map):,}개 이어서 시작")

    done = set(llr_map.keys())
    llrs = list(llr_map.items())
    batch_seqs, batch_refs, batch_alts, batch_idx = [], [], [], []
    processed = len(done)

    def flush():
        if not batch_seqs:
            return
        scores = compute_llr_batch(batch_seqs, batch_refs, batch_alts,
                                   tokenizer, model, device)
        llrs.extend(zip(batch_idx, scores))
        batch_seqs.clear(); batch_refs.clear()
        batch_alts.clear(); batch_idx.clear()

    def save_checkpoint():
        pd.DataFrame(llrs, columns=["idx", "llr"]).to_parquet(ckpt_path, index=False)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="PhyloGPN VEP"):
        if i in done:
            continue

        seq = extract_seq(genome, str(row.chrom), int(row.pos), WINDOW)
        if seq is None or len(seq) != WINDOW:
            llrs.append((i, None))
        else:
            batch_seqs.append(seq)
            batch_refs.append(row.ref)
            batch_alts.append(row.alt)
            batch_idx.append(i)

            if len(batch_seqs) >= args.batch_size:
                flush()

        processed += 1
        if processed % CHECKPOINT_INTERVAL == 0:
            flush()
            save_checkpoint()

    flush()

    llr_map = {idx: score for idx, score in llrs}
    df["llr_phylogpn"] = [llr_map.get(i) for i in df.index]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    valid = df["llr_phylogpn"].notna().sum()
    print(f"\n완료: {valid:,}/{len(df):,} 변이 스코어링됨")
    print(f"저장: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",   required=True, help="입력 parquet (chrom,pos,ref,alt,...)")
    parser.add_argument("--genome",     required=True, help="hg38.fa (pyfaidx 인덱스 필요)")
    parser.add_argument("--out",        required=True, help="출력 parquet")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run_vep(args)


if __name__ == "__main__":
    main()
