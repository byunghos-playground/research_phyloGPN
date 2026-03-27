"""
train_f81_supervised.py

모델 C 훈련: F81 Supervised (likelihood matching).

[훈련 방식]
  F81 (unsupervised) 과 Naive (직접 supervised) 사이의 중간:
  - 모델은 ref_seq 481bp만 input으로 받음 (F81과 동일)
  - loss = log P_F81(alignment | π_true, T) - log P_F81(alignment | π_pred, T)
  - π_true를 직접 비교하지 않고, likelihood를 통해 간접적으로 활용

[F81 vs f81_supervised vs naive 비교]
  F81:            loss = -log P_F81(alignment | π_pred, T)           ← π_true 없음
  f81_supervised: loss = log P_F81(alignment | π_true, T)
                       - log P_F81(alignment | π_pred, T)            ← π_true 간접
  Naive:          loss = KL(π_true || π_pred)                        ← π_true 직접

Usage:
  python train_f81_supervised.py \\
    --train_dir   data/train \\
    --valid_dir   data/valid \\
    --tree_path   data/trees/241-mammalian-2020v2.1.nh.txt \\
    --out_dir     checkpoints/f81_supervised \\
    --batch_size  8 \\
    --epochs      20 \\
    --lr          1e-4 \\
    --stride      1
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration   import PhyloGPNConfig
from src.models.model            import PhyloGPNModel
from src.models.tokenizer        import PhyloGPNTokenizer
from src.data.windowed_dataset   import WindowedSimF81Dataset
from src.data.collate            import collate_windowed_sim_f81
from src.losses.f81_supervised_loss import F81SupervisedLoss
from src.utils.tree_utils        import load_tree_struct_from_newick
from src.utils.checkpoint        import save_checkpoint, BestModelTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhyloGPN 훈련 (F81 supervised loss).")
    p.add_argument("--train_dir",   type=str, required=True)
    p.add_argument("--valid_dir",   type=str, default=None)
    p.add_argument("--tree_path",   type=str, required=True)
    p.add_argument("--out_dir",     type=str, default="checkpoints/f81_supervised")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--stride",      type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--window_size", type=int, default=481)
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
    # ------------------------------------------------------------------
    train_paths = find_npz(args.train_dir)
    first_npz   = np.load(train_paths[0], allow_pickle=True)
    if "taxon_names" not in first_npz:
        raise RuntimeError(f"'{train_paths[0]}' 에 'taxon_names' 없음.")
    leaf_order  = list(map(str, first_npz["taxon_names"]))
    tree_struct = load_tree_struct_from_newick(args.tree_path, leaf_order)
    print(f"계통수 로드 완료: {tree_struct.n_nodes} 노드, {tree_struct.n_leaves} leaf")

    # ------------------------------------------------------------------
    # 3. Dataset / DataLoader
    #    use_msa=True: F81 likelihood 계산에 alignment 필요
    #    pi_true도 배치에 포함됨 (WindowedSimF81Dataset 기본 포함)
    # ------------------------------------------------------------------
    train_ds = WindowedSimF81Dataset(
        npz_paths   = train_paths,
        tokenizer   = tokenizer,
        window_size = args.window_size,
        use_msa     = True,
        stride      = args.stride,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_windowed_sim_f81,
        pin_memory  = (args.device == "cuda"),
    )

    valid_loader = None
    if args.valid_dir:
        valid_ds = WindowedSimF81Dataset(
            npz_paths   = find_npz(args.valid_dir),
            tokenizer   = tokenizer,
            window_size = args.window_size,
            use_msa     = True,
            stride      = args.stride * 5,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = collate_windowed_sim_f81,
        )

    print(f"훈련 윈도우 수: {len(train_ds):,}")

    # ------------------------------------------------------------------
    # 4. 모델
    # ------------------------------------------------------------------
    cfg   = PhyloGPNConfig()
    model = PhyloGPNModel(cfg).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 5. Loss & Optimizer
    # ------------------------------------------------------------------
    loss_fn   = F81SupervisedLoss(tree_struct=tree_struct)
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
            input_ids  = batch["input_ids"].to(device)    # (B, W)
            msa_codes  = batch["msa_codes"].to(device)    # (B, W, S)
            pi_true    = batch["pi_true"].to(device)      # (B, W, 4)
            center_idx = batch["center_idx"].to(device)   # (B,)
            B          = input_ids.shape[0]

            optimizer.zero_grad()
            logits_dict = model(input_ids)   # {'A','C','G','T'}: (B, 1)

            # center column만 loss에 사용
            c              = center_idx[0].item()           # 항상 240
            msa_center     = msa_codes[:, c:c+1, :]         # (B, 1, S)
            pi_true_center = pi_true[:, c:c+1, :]           # (B, 1, 4)
            valid_center   = torch.ones(B, 1, dtype=torch.bool, device=device)

            loss = loss_fn(
                logits_dict = logits_dict,
                msa_codes   = msa_center,
                pi_true     = pi_true_center,
                valid_mask  = valid_center,
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
            val_loss = 0.0
            val_n    = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids  = batch["input_ids"].to(device)
                    msa_codes  = batch["msa_codes"].to(device)
                    pi_true    = batch["pi_true"].to(device)
                    center_idx = batch["center_idx"].to(device)
                    B          = input_ids.shape[0]

                    logits_dict    = model(input_ids)
                    c              = center_idx[0].item()
                    msa_center     = msa_codes[:, c:c+1, :]
                    pi_true_center = pi_true[:, c:c+1, :]
                    valid_center   = torch.ones(B, 1, dtype=torch.bool, device=device)

                    loss = loss_fn(logits_dict, msa_center, pi_true_center, valid_center)
                    val_loss += loss.item()
                    val_n    += 1

            avg_valid = val_loss / max(1, val_n)
            tracker.update(avg_valid, model, optimizer, epoch,
                           config=cfg.__dict__)

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train={avg_train:.5f}  valid={avg_valid:.5f}")

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
