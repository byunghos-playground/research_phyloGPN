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
    # 4. 지표 계산 & 저장
    # ------------------------------------------------------------------
    metrics = compute_metrics(pi_pred_all, pi_true_all)
    n_sites = len(pi_pred_all)

    lines = [
        f"=== {args.model_name} 평가 결과 ===",
        f"평가 사이트 수: {n_sites:,}",
        "",
        "MAE (낮을수록 좋음)",
        f"  A : {metrics['mae_A']:.4f}",
        f"  C : {metrics['mae_C']:.4f}",
        f"  G : {metrics['mae_G']:.4f}",
        f"  T : {metrics['mae_T']:.4f}",
        f"  전체 평균: {metrics['mae_mean']:.4f}",
        "",
        "Pearson r (높을수록 좋음, 최대 1.0)",
        f"  A : {metrics['pearson_A']:.4f}",
        f"  C : {metrics['pearson_C']:.4f}",
        f"  G : {metrics['pearson_G']:.4f}",
        f"  T : {metrics['pearson_T']:.4f}",
        f"  전체 (4N 쌍): {metrics['pearson_all']:.4f}",
        "",
        f"KL(π_true || π_pred) 평균: {metrics['kl_mean']:.6f}  (낮을수록 좋음)",
        "",
        "─" * 60,
        "[데이터 및 평가 방법]",
        "  - 실험: GC OU process 기반 position-varying π 시뮬레이션 (exp3/4)",
        "  - 입력: genome 10,000bp에서 stride=481 sliding window로 뽑은 481bp 구간",
        "  - 모델 출력: 481bp → valid conv (RF=481) → center 1 position의 π 예측",
        "  - 비교: π_pred (모델 예측) vs π_true (시뮬레이션에 사용된 실제 stationary freq)",
        "",
        "[각 지표 의미]",
        "  MAE        : 예측한 각 염기 빈도와 실제 빈도의 절대 오차 평균.",
        "               0에 가까울수록 정확. 무작위 예측 시 약 0.125.",
        "  Pearson r  : π_pred와 π_true 간 선형 상관계수.",
        "               1.0이면 완벽한 양의 상관, 0이면 무관계.",
        "               모델이 π의 상대적 변화를 잘 추적하는지 반영.",
        "  KL divergence: KL(π_true || π_pred). π_true 분포 관점에서",
        "               π_pred가 얼마나 다른지. 0이면 완벽 일치.",
    ]

    report = "\n".join(lines)
    print(report)

    out_path = os.path.join(args.out_dir, f"eval_{args.model_name}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
