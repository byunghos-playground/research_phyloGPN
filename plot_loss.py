"""
plot_loss.py

exp3~6 훈련 loss curve 시각화.
checkpoints/{exp}/{model}/train.log 파싱 → PNG 저장.

Usage:
  python plot_loss.py [--out_dir results/loss_curves] [--exps exp3_gc exp4_gc_r ...]
"""

import argparse
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# log line 패턴: [Epoch  3/20] train=0.12345  valid=0.12345
LOG_RE = re.compile(
    r"\[Epoch\s+(\d+)/\d+\]\s+train=([\d.naninf]+)\s+valid=([\d.naninf]+)"
)

MODELS = ["f81", "f81_supervised", "naive"]
MODEL_LABELS = {"f81": "F81", "f81_supervised": "F81-Supervised", "naive": "Naive"}
MODEL_COLORS = {"f81": "#1f77b4", "f81_supervised": "#ff7f0e", "naive": "#2ca02c"}
MODEL_YLABELS = {
    "f81":            "-log P_F81(aln|π_pred,T) + log π_ref",
    "f81_supervised": "log P_F81(aln|π_true,T) - log P_F81(aln|π_pred,T)",
    "naive":          "KL(π_true || π_pred)",
}


def parse_log(log_path):
    """train.log → {epoch: int, train: float, valid: float} 리스트.

    재훈련으로 log가 이어붙여진 경우 epoch 번호가 리셋되면
    새 run으로 간주하고 마지막 run만 반환.
    """
    if not os.path.exists(log_path):
        return []

    runs = []
    current = []
    with open(log_path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            train = float(m.group(2)) if m.group(2) not in ("nan", "inf") else float("nan")
            valid = float(m.group(3)) if m.group(3) not in ("nan", "inf") else float("nan")
            # epoch가 리셋되면 (새 run 시작)
            if current and epoch <= current[-1]["epoch"]:
                runs.append(current)
                current = []
            current.append({"epoch": epoch, "train": train, "valid": valid})
    if current:
        runs.append(current)

    if not runs:
        return []

    if len(runs) > 1:
        print(f"  [INFO] {os.path.basename(os.path.dirname(log_path))}/train.log: "
              f"{len(runs)} runs 감지, 마지막 run ({len(runs[-1])} epochs) 사용")
    return runs[-1]


def plot_exp(exp, ckpt_root, out_dir):
    """한 실험의 3개 모델 loss curve — 모델별 row (별도 y축), train/valid col."""
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(12, 4 * len(MODELS)))
    fig.suptitle(f"{exp}  —  Loss Curves", fontsize=14)

    for row, model in enumerate(MODELS):
        log_path = os.path.join(ckpt_root, exp, model, "train.log")
        records  = parse_log(log_path)
        color    = MODEL_COLORS[model]
        label    = MODEL_LABELS[model]
        ylabel   = MODEL_YLABELS[model]

        for col, split in enumerate(["train", "valid"]):
            ax = axes[row, col]
            ax.set_title(f"{label}  —  {split.capitalize()}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel, fontsize=7)
            ax.grid(True, alpha=0.3)

            if not records:
                ax.text(0.5, 0.5, "log 없음", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue

            epochs = [r["epoch"] for r in records]
            vals   = [r[split]   for r in records]
            ax.plot(epochs, vals, color=color, marker="o", markersize=3)

        if records:
            trains = [r["train"] for r in records]
            valids = [r["valid"] for r in records]
            print(f"  {exp}/{model}: {len(records)} epochs, "
                  f"final train={trains[-1]:.5f}, valid={valids[-1]:.5f}")
        else:
            print(f"  [SKIP] {log_path} 없음 또는 비어있음")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"loss_{exp}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → 저장: {out_path}")


def plot_all_exps(exps, ckpt_root, out_dir):
    """모든 실험을 한 figure에 (모델별 subrow)."""
    n_exps   = len(exps)
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
    fig.suptitle("All Experiments — Train / Valid Loss", fontsize=14)

    for row, model in enumerate(MODELS):
        for col, split in enumerate(["train", "valid"]):
            ax = axes[row, col]
            ax.set_title(f"{MODEL_LABELS[model]}  —  {split.capitalize()} Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(MODEL_YLABELS[model], fontsize=7)
            ax.grid(True, alpha=0.3)

            for exp in exps:
                log_path = os.path.join(ckpt_root, exp, model, "train.log")
                records  = parse_log(log_path)
                if not records:
                    continue
                epochs = [r["epoch"] for r in records]
                vals   = [r[split]  for r in records]
                ax.plot(epochs, vals, label=exp, marker="o", markersize=3)

            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "loss_all_exps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"전체 비교 → 저장: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default="checkpoints",
                   help="checkpoints 루트 디렉토리")
    p.add_argument("--out_dir",   default="results/loss_curves")
    p.add_argument("--exps", nargs="+",
                   default=["exp3_gc", "exp4_gc_r", "exp5_gc", "exp6_gc_r"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for exp in args.exps:
        print(f"\n=== {exp} ===")
        plot_exp(exp, args.ckpt_root, args.out_dir)

    print("\n=== 전체 비교 ===")
    plot_all_exps(args.exps, args.ckpt_root, args.out_dir)

    print("\n완료.")


if __name__ == "__main__":
    main()
