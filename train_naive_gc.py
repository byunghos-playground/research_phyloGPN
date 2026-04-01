"""
train_naive_gc.py

[Exp3 / Exp4] 모델 Naive 훈련 — GC continuous variation 데이터용.

[baseline train_naive.py와의 차이]
  - WindowedSimF81Dataset 사용: 긴 genome npz → sliding window로 분해
  - π_true는 position마다 다름 (GC OU process 기반 continuous variation)

Usage:
  python train_naive_gc.py \\
    --data_dir    data/exp3_gc/processed \\
    --out_dir     checkpoints/exp3_gc/naive \\
    --stride      1
"""

import argparse
import glob
import logging
import os
import random
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration   import PhyloGPNConfig
from src.models.model           import PhyloGPNModel
from src.models.tokenizer       import PhyloGPNTokenizer
from src.data.windowed_dataset  import WindowedSimF81Dataset
from src.data.collate           import collate_sim_f81
from src.losses.supervised_loss import SupervisedPiLoss
from src.utils.checkpoint       import save_checkpoint, BestModelTracker
from src.utils.math_f81         import logits_dict_to_pi


def parse_args():
    p = argparse.ArgumentParser(description="PhyloGPN naive supervised 훈련 (GC sliding window).")
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--valid_ratio",  type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out_dir",      type=str, default="checkpoints/exp3_gc/naive")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--stride",       type=int, default=1)
    p.add_argument("--window_size",  type=int, default=481)
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",       type=str, default=None)
    return p.parse_args()


def split_npz(data_dir, train_ratio, valid_ratio, seed):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise RuntimeError(f"'{data_dir}' 에 .npz 파일이 없습니다.")
    random.seed(seed)
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    return paths[:n_train], paths[n_train:n_train + n_valid]


def setup_logger(out_dir):
    logger = logging.getLogger("train_naive_gc")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    log = setup_logger(args.out_dir)

    tokenizer = PhyloGPNTokenizer(model_max_length=10 ** 9)

    train_paths, valid_paths = split_npz(
        args.data_dir, args.train_ratio, args.valid_ratio, args.seed
    )
    log.info(f"데이터 split: train={len(train_paths)} genomes, valid={len(valid_paths)} genomes")

    train_ds = WindowedSimF81Dataset(
        npz_paths=train_paths, tokenizer=tokenizer,
        window_size=args.window_size, use_msa=False, stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_sim_f81,
        pin_memory=(args.device == "cuda"),
    )

    valid_loader = None
    if valid_paths:
        valid_ds = WindowedSimF81Dataset(
            npz_paths=valid_paths, tokenizer=tokenizer,
            window_size=args.window_size, use_msa=False, stride=args.stride,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_sim_f81,
        )

    log.info(f"훈련 window 수: {len(train_ds):,}")

    cfg   = PhyloGPNConfig()
    model = PhyloGPNModel(cfg).to(device)
    log.info(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn   = SupervisedPiLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        from src.utils.checkpoint import load_checkpoint
        info        = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = info["epoch"] + 1

    tracker = BestModelTracker(os.path.join(args.out_dir, "best.pt"))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            input_ids  = batch["input_ids"].to(device)
            pi_true    = batch["pi_true"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            logits_dict = model(input_ids)
            pi_pred = logits_dict_to_pi(logits_dict)
            loss = loss_fn(pi_pred=pi_pred, pi_true=pi_true, valid_mask=valid_mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        avg_train = train_loss / max(1, n_batches)

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
                    pi_pred = logits_dict_to_pi(logits_dict)
                    loss = loss_fn(pi_pred, pi_true, valid_mask)
                    val_loss += loss.item()
                    val_n    += 1
            avg_valid = val_loss / max(1, val_n)
            tracker.update(avg_valid, model, optimizer, epoch, config=cfg.__dict__)

        log.info(f"[Epoch {epoch:3d}/{args.epochs}] train={avg_train:.5f}  valid={avg_valid:.5f}")
        save_checkpoint(
            path=os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
            model=model, optimizer=optimizer, epoch=epoch,
            loss=avg_train, config=cfg.__dict__,
        )

    log.info("훈련 완료.")


if __name__ == "__main__":
    main()
