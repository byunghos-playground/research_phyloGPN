"""
train_naive.py

모델 B 훈련: Supervised (naive) baseline.

[훈련 방식]
  - F81 loss 없음, tree 없음, alignment 없음
  - 모델이 ref_seq만 보고 π_true (ground-truth F81 파라미터)를 직접 예측
  - Loss: KL(π_true || π_pred)
  - SimF81Dataset (블록 단위, 윈도우 없음) 사용

[F81 loss 모델과의 차이]
  F81 모델: alignment y와 tree T를 통해 간접 학습 (비지도적)
  Naive  모델: π_true를 label로 직접 학습 (지도 학습)

  Naive 모델의 성능이 F81 모델보다 나쁘면?
    → alignment 정보 없이 ref_seq 맥락만으로는 진화적 constraint를 학습하기 어려움
  F81 모델이 더 나쁘면?
    → F81 framework나 구현에 문제가 있을 가능성

[주의]
  Naive 학습은 실제 게놈 데이터에는 적용 불가 (π_true를 알 수 없음).
  오직 simulation study 맥락에서만 의미있음.

Usage:
  python train_naive.py \\
    --train_dir  data/train \\
    --valid_dir  data/valid \\
    --out_dir    checkpoints/naive \\
    --batch_size 4 \\
    --epochs     20 \\
    --lr         1e-4
"""

import argparse
import glob
import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration import PhyloGPNConfig
from src.models.model         import PhyloGPNModel
from src.models.tokenizer     import PhyloGPNTokenizer
from src.data.dataset         import SimF81Dataset
from src.data.collate         import collate_sim_f81
from src.losses.supervised_loss import SupervisedPiLoss
from src.utils.checkpoint     import save_checkpoint, BestModelTracker
from src.utils.math_f81       import logits_dict_to_pi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhyloGPN naive supervised 훈련.")
    p.add_argument("--train_dir",   type=str, required=True)
    p.add_argument("--valid_dir",   type=str, default=None)
    p.add_argument("--out_dir",     type=str, default="checkpoints/naive")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--pad_half",    type=int, default=240,
                   help="ref_seq 양쪽 패딩 길이 (기본값 240 = RF//2)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",      type=str, default=None)
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
    # 1. Tokenizer & Dataset
    # ------------------------------------------------------------------
    tokenizer = PhyloGPNTokenizer(model_max_length=10 ** 9)

    train_ds = SimF81Dataset(
        npz_paths = find_npz(args.train_dir),
        tokenizer = tokenizer,
        pad_half  = args.pad_half,
        use_msa   = False,   # supervised: alignment 불필요
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
            pad_half  = args.pad_half,
            use_msa   = False,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = collate_sim_f81,
        )

    print(f"훈련 블록 수: {len(train_ds)}")

    # ------------------------------------------------------------------
    # 2. 모델 & Loss & Optimizer
    # ------------------------------------------------------------------
    cfg   = PhyloGPNConfig()
    model = PhyloGPNModel(cfg).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn   = SupervisedPiLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        from src.utils.checkpoint import load_checkpoint
        info        = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = info["epoch"] + 1

    tracker = BestModelTracker(os.path.join(args.out_dir, "best.pt"))

    # ------------------------------------------------------------------
    # 3. 훈련 루프
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            input_ids  = batch["input_ids"].to(device)    # (B, Lp)
            pi_true    = batch["pi_true"].to(device)      # (B, L, 4)
            valid_mask = batch["valid_mask"].to(device)   # (B, L)

            optimizer.zero_grad()
            logits_dict = model(input_ids)   # {'A','C','G','T'}: (B, L_out)

            # logits → π (B, L_out, 4)
            pi_pred = logits_dict_to_pi(logits_dict)   # (B, L_out, 4)

            # L_out ≠ L 이면 중앙 crop
            B, L, _ = pi_true.shape
            L_out   = pi_pred.shape[1]
            if L_out != L:
                crop    = (L_out - L) // 2
                pi_pred = pi_pred[:, crop: crop + L, :]

            loss = loss_fn(pi_pred, pi_true, valid_mask)
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
                    pi_true    = batch["pi_true"].to(device)
                    valid_mask = batch["valid_mask"].to(device)

                    logits_dict = model(input_ids)
                    pi_pred     = logits_dict_to_pi(logits_dict)

                    B, L, _ = pi_true.shape
                    L_out   = pi_pred.shape[1]
                    if L_out != L:
                        crop    = (L_out - L) // 2
                        pi_pred = pi_pred[:, crop: crop + L, :]

                    loss    = loss_fn(pi_pred, pi_true, valid_mask)
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
