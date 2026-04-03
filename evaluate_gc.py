"""
evaluate_gc.py

[Exp3 / Exp4] 모델 평가 — GC continuous variation 데이터용.
WindowedSimF81Dataset으로 center-site π_pred vs π_true 비교.

[exp1/2 evaluate.py와의 차이]
  - WindowedSimF81Dataset 사용 (긴 genome npz → sliding window)
  - 모델 출력 L_out=1 (입력 481bp, RF=481 → center 1 position만 출력)
  - split.json에서 test paths 읽음

[평가 지표]
  1. MAE  : |π_pred - π_true| 평균 (각 염기별 + 전체)
  2. Pearson r: π_pred vs π_true 선형 상관 (각 염기별 + 전체)
  3. KL   : KL(π_true || π_pred) 평균

Usage:
  python evaluate_gc.py \\
    --checkpoint checkpoints/exp3_gc/f81/best.pt \\
    --split_json checkpoints/exp3_gc/f81/split.json \\
    --tree_path  data/trees/241-mammalian-2020v2.1.nh.txt \\
    --model_name exp3_f81 \\
    --stride 481
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration   import PhyloGPNConfig
from src.models.model           import PhyloGPNModel
from src.models.tokenizer       import PhyloGPNTokenizer
from src.data.windowed_dataset  import WindowedSimF81Dataset
from src.data.collate           import collate_windowed_sim_f81
from src.utils.math_f81         import logits_dict_to_pi
from src.utils.checkpoint       import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="[Exp3/4] 모델 평가: center-site π_pred vs π_true.")
    p.add_argument("--checkpoint",    type=str, required=True,  help="체크포인트 경로")
    p.add_argument("--split_json",    type=str, required=True,  help="split.json 경로")
    p.add_argument("--model_name",    type=str, default="model", help="결과 파일 이름용")
    p.add_argument("--out_dir",       type=str, default="results")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--stride",        type=int, default=481,    help="평가 sliding window 보폭 (기본: 비겹침)")
    p.add_argument("--window_size",   type=int, default=481)
    p.add_argument("--block_length",  type=int, default=10000,  help="genome 고정 길이 (on-demand 로딩)")
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def compute_metrics(
    pi_pred_all: np.ndarray,   # (N, 4)
    pi_true_all: np.ndarray,   # (N, 4)
) -> Dict[str, float]:
    bases  = ["A", "C", "G", "T"]
    result = {}

    mae = np.abs(pi_pred_all - pi_true_all).mean(axis=0)   # (4,)
    for i, b in enumerate(bases):
        result[f"mae_{b}"] = float(mae[i])
    result["mae_mean"] = float(mae.mean())

    for i, b in enumerate(bases):
        r, _ = stats.pearsonr(pi_pred_all[:, i], pi_true_all[:, i])
        result[f"pearson_{b}"] = float(r)

    r_all, _ = stats.pearsonr(pi_pred_all.ravel(), pi_true_all.ravel())
    result["pearson_all"] = float(r_all)

    eps  = 1e-8
    pt_c = np.clip(pi_true_all, eps, None)
    pp_c = np.clip(pi_pred_all, eps, None)
    kl   = (pt_c * (np.log(pt_c) - np.log(pp_c))).sum(axis=-1)   # (N,)
    result["kl_mean"] = float(kl.mean())

    return result


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 모델 로드
    # ------------------------------------------------------------------
    cfg   = PhyloGPNConfig()
    model = PhyloGPNModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()
    print(f"모델 로드: {args.checkpoint}")

    # ------------------------------------------------------------------
    # 2. 테스트 데이터 (split.json의 test 경로)
    # ------------------------------------------------------------------
    with open(args.split_json) as f:
        test_paths = json.load(f)["test"]
    if not test_paths:
        raise RuntimeError(f"'{args.split_json}' 의 test 목록이 비어 있음.")
    print(f"테스트 genome 수: {len(test_paths)}")

    tokenizer = PhyloGPNTokenizer(model_max_length=10 ** 9)
    test_ds = WindowedSimF81Dataset(
        npz_paths    = test_paths,
        tokenizer    = tokenizer,
        window_size  = args.window_size,
        use_msa      = False,
        stride       = args.stride,
        cache        = False,
        block_length = args.block_length,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_windowed_sim_f81,
    )
    print(f"평가 window 수: {len(test_ds):,}  (stride={args.stride})")

    # ------------------------------------------------------------------
    # 3. 예측 — center 1 position (L_out=1)
    # ------------------------------------------------------------------
    pi_pred_list = []
    pi_true_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)   # (B, W)
            c         = batch["center_idx"][0].item()   # 항상 240

            logits_dict = model(input_ids)
            pi_pred     = logits_dict_to_pi(logits_dict).cpu()   # (B, L_out, 4)

            # L_out=1: squeeze to (B, 4)
            pi_pred_c = pi_pred[:, 0, :].numpy()                 # (B, 4)
            pi_true_c = batch["pi_true"][:, c, :].numpy()        # (B, 4)

            pi_pred_list.append(pi_pred_c)
            pi_true_list.append(pi_true_c)

    pi_pred_all = np.concatenate(pi_pred_list, axis=0)   # (N, 4)
    pi_true_all = np.concatenate(pi_true_list, axis=0)
    print(f"총 평가 사이트 수: {len(pi_pred_all):,}")

    # ------------------------------------------------------------------
    # 4. 지표 계산 & 출력
    # ------------------------------------------------------------------
    metrics = compute_metrics(pi_pred_all, pi_true_all)

    print("\n=== 평가 결과 ===")
    print(f"  MAE (A/C/G/T): "
          f"{metrics['mae_A']:.4f} / {metrics['mae_C']:.4f} / "
          f"{metrics['mae_G']:.4f} / {metrics['mae_T']:.4f}")
    print(f"  MAE mean     : {metrics['mae_mean']:.4f}")
    print(f"  Pearson r (A/C/G/T): "
          f"{metrics['pearson_A']:.4f} / {metrics['pearson_C']:.4f} / "
          f"{metrics['pearson_G']:.4f} / {metrics['pearson_T']:.4f}")
    print(f"  Pearson r (all)    : {metrics['pearson_all']:.4f}")
    print(f"  KL(true||pred) mean: {metrics['kl_mean']:.6f}")

    # ------------------------------------------------------------------
    # 5. JSON 저장
    # ------------------------------------------------------------------
    out_path = os.path.join(args.out_dir, f"eval_{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
