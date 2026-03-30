"""
train_f81.py

모델 A 훈련: F81 phylogenetic likelihood loss.

[훈련 방식]
  - SimF81Dataset: 청크당 π 하나, L=481 사이트 (pad_half=240으로 패딩)
  - All-site loss: 청크 내 모든 L개 위치에 F81 loss 계산 (valid_mask 기준)
    → 같은 π에서 생성된 481개 사이트 전체에서 gradient 신호 획득
  - Felsenstein pruning으로 alignment column의 log-likelihood 계산

[중요]
  F81 loss는 msa_codes(alignment)와 tree가 필요.
  msa_codes는 .npz의 'msa_codes' 키에서 로드.

Usage:
  python train_f81.py \\
    --train_dir   data/train \\
    --valid_dir   data/valid \\
    --tree_path   data/trees/241-mammalian-2020v2.1.nh.txt \\
    --out_dir     checkpoints/f81 \\
    --batch_size  8 \\
    --epochs      20 \\
    --lr          1e-4
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# src 패키지를 python path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration import PhyloGPNConfig
from src.models.model         import PhyloGPNModel
from src.models.tokenizer     import PhyloGPNTokenizer
from src.data.dataset          import SimF81Dataset
from src.data.collate          import collate_sim_f81
from src.losses.f81_loss       import F81LikelihoodLoss
from src.utils.tree_utils      import load_tree_struct_from_newick
from src.utils.checkpoint      import save_checkpoint, BestModelTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhyloGPN 훈련 (F81 loss).")
    p.add_argument("--train_dir",  type=str, required=True,  help="훈련 .npz 디렉토리")
    p.add_argument("--valid_dir",  type=str, default=None,   help="검증 .npz 디렉토리 (없으면 skip)")
    p.add_argument("--tree_path",  type=str, required=True,  help="Newick 계통수 파일")
    p.add_argument("--out_dir",    type=str, default="checkpoints/f81")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",     type=str, default=None,
                   help="재개할 체크포인트 경로")
    return p.parse_args()


def find_npz(directory: str):
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise RuntimeError(f"'{directory}' 에 .npz 파일이 없습니다.")
    return paths


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = PhyloGPNTokenizer(model_max_length=10 ** 9)

    # ------------------------------------------------------------------
    # 2. 계통수 로드
    #    taxon_names는 첫 번째 npz에서 읽어서 leaf_order 결정
    # ------------------------------------------------------------------
    train_paths = find_npz(args.train_dir)
    first_npz   = np.load(train_paths[0], allow_pickle=True)
    if "taxon_names" not in first_npz:
        raise RuntimeError(
            f"'{train_paths[0]}' 에 'taxon_names' 없음. "
            "fasta_to_npz.py로 변환 시 taxon_names가 저장되어야 합니다."
        )
    leaf_order = list(map(str, first_npz["taxon_names"]))
    tree_struct = load_tree_struct_from_newick(args.tree_path, leaf_order)
    print(f"계통수 로드 완료: {tree_struct.n_nodes} 노드, {tree_struct.n_leaves} leaf")

    # ------------------------------------------------------------------
    # 3. Dataset / DataLoader
    # ------------------------------------------------------------------
    train_ds = SimF81Dataset(
        npz_paths = train_paths,
        tokenizer = tokenizer,
        pad_half  = 240,
        use_msa   = True,      # F81 loss에 alignment 필요
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_sim_f81,
        pin_memory  = (args.device == "cuda"),
    )

    valid_loader = None
    if args.valid_dir:
        valid_ds = SimF81Dataset(
            npz_paths = find_npz(args.valid_dir),
            tokenizer = tokenizer,
            pad_half  = 240,
            use_msa   = True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = collate_sim_f81,
        )

    print(f"훈련 청크 수: {len(train_ds):,}")

    # ------------------------------------------------------------------
    # 4. 모델
    # ------------------------------------------------------------------
    cfg   = PhyloGPNConfig()
    model = PhyloGPNModel(cfg).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 5. Loss & Optimizer
    # ------------------------------------------------------------------
    loss_fn   = F81LikelihoodLoss(tree_struct=tree_struct)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        from src.utils.checkpoint import load_checkpoint
        info        = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = info["epoch"] + 1

    tracker = BestModelTracker(os.path.join(args.out_dir, "best.pt"))

    # ------------------------------------------------------------------
    # 6. 훈련 루프
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            input_ids  = batch["input_ids"].to(device)    # (B, L+480)
            msa_codes  = batch["msa_codes"].to(device)    # (B, L, S)
            valid_mask = batch["valid_mask"].to(device)   # (B, L)

            optimizer.zero_grad()
            logits_dict = model(input_ids)   # {'A','C','G','T'}: (B, L)

            loss = loss_fn(
                logits_dict = logits_dict,
                msa_codes   = msa_codes,
                valid_mask  = valid_mask,
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        avg_train = train_loss / max(1, n_batches)

        # --- Validation ---
        avg_valid = float("nan")
        if valid_loader is not None:
            model.eval()
            val_loss  = 0.0
            val_n     = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids  = batch["input_ids"].to(device)
                    msa_codes  = batch["msa_codes"].to(device)
                    valid_mask = batch["valid_mask"].to(device)

                    logits_dict = model(input_ids)
                    loss = loss_fn(logits_dict, msa_codes, valid_mask)
                    val_loss += loss.item()
                    val_n    += 1

            avg_valid = val_loss / max(1, val_n)
            tracker.update(avg_valid, model, optimizer, epoch,
                           config=cfg.__dict__)

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train={avg_train:.5f}  valid={avg_valid:.5f}")

        # epoch 체크포인트 저장
        save_checkpoint(
            path      = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
            model     = model,
            optimizer = optimizer,
            epoch     = epoch,
            loss      = avg_train,
            config    = cfg.__dict__,
        )

    print("훈련 완료.")


if __name__ == "__main__":
    main()
