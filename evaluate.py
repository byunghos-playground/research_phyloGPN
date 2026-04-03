"""
evaluate.py

모델 평가: 예측된 π vs ground-truth π_true 비교.

[평가 지표]
  1. MAE (Mean Absolute Error): |π_pred - π_true| 평균
     → 각 염기 (A/C/G/T)별, 전체 평균
  2. Pearson correlation: π_pred와 π_true 사이 선형 상관
     → 각 염기별, 전체 (all 4 * n_sites 쌍)
  3. KL divergence: KL(π_true || π_pred) 평균
     → supervised loss와 동일 지표

[F81 framework 검증 의미]
  - F81 모델이 잘 훈련되었다면 π_pred ≈ π_true 여야 함
  - Naive 모델과 비교:
      * F81 > Naive: phylogenetic signal이 F81 framework를 통해 잘 전달됨
      * F81 ≈ Naive: alignment 없이도 ref_seq 맥락만으로 충분
      * F81 < Naive: F81 구현에 문제가 있을 가능성

[출력]
  - 콘솔: 각 지표 요약
  - results/eval_{model_name}.json: 전체 지표 JSON 저장

Usage:
  # F81 모델 평가
  python evaluate.py \\
    --checkpoint checkpoints/f81/best.pt \\
    --test_dir   data/test \\
    --tree_path  data/trees/241-mammalian-2020v2.1.nh.txt \\
    --model_name f81

  # Naive 모델 평가
  python evaluate.py \\
    --checkpoint checkpoints/naive/best.pt \\
    --test_dir   data/test \\
    --model_name naive
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats   # pip install scipy

sys.path.insert(0, os.path.dirname(__file__))

from src.models.configuration  import PhyloGPNConfig
from src.models.model          import PhyloGPNModel
from src.models.tokenizer      import PhyloGPNTokenizer
from src.data.dataset          import SimF81Dataset
from src.data.collate          import collate_sim_f81
from src.utils.math_f81        import logits_dict_to_pi
from src.utils.checkpoint      import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모델 평가: π_pred vs π_true.")
    p.add_argument("--checkpoint",  type=str, required=True,  help="체크포인트 경로")
    p.add_argument("--test_dir",    type=str, default=None,   help="테스트 .npz 디렉토리")
    p.add_argument("--split_json",  type=str, default=None,   help="split.json 경로 (test 키에서 경로 읽기)")
    p.add_argument("--model_name",  type=str, default="model", help="결과 파일 이름용")
    p.add_argument("--out_dir",     type=str, default="results")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--pad_half",    type=int, default=240)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def compute_metrics(
    pi_pred_all: np.ndarray,   # (N, 4)
    pi_true_all: np.ndarray,   # (N, 4)
) -> Dict[str, float]:
    """
    N개 사이트에 대한 평가 지표 계산.

    Returns
    -------
    dict with:
      mae_A/C/G/T        : 각 염기별 MAE
      mae_mean           : 전체 MAE
      pearson_A/C/G/T    : 각 염기별 Pearson r
      pearson_all        : 전체 (4N) Pearson r
      kl_mean            : 평균 KL(π_true || π_pred)
    """
    bases  = ["A", "C", "G", "T"]
    result = {}

    # MAE per base
    mae = np.abs(pi_pred_all - pi_true_all).mean(axis=0)  # (4,)
    for i, b in enumerate(bases):
        result[f"mae_{b}"] = float(mae[i])
    result["mae_mean"] = float(mae.mean())

    # Pearson per base
    for i, b in enumerate(bases):
        r, _ = stats.pearsonr(pi_pred_all[:, i], pi_true_all[:, i])
        result[f"pearson_{b}"] = float(r)

    # Pearson (all 4N pairs)
    r_all, _ = stats.pearsonr(
        pi_pred_all.ravel(), pi_true_all.ravel()
    )
    result["pearson_all"] = float(r_all)

    # KL divergence KL(π_true || π_pred)
    eps  = 1e-8
    pt_c = np.clip(pi_true_all,  eps, None)
    pp_c = np.clip(pi_pred_all,  eps, None)
    kl   = (pt_c * (np.log(pt_c) - np.log(pp_c))).sum(axis=-1)  # (N,)
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
    # 2. 테스트 데이터
    # ------------------------------------------------------------------
    tokenizer = PhyloGPNTokenizer(model_max_length=10 ** 9)
    if args.split_json:
        with open(args.split_json) as f:
            npz_paths = json.load(f)["test"]
        if not npz_paths:
            raise RuntimeError(f"'{args.split_json}' 의 test 목록이 비어 있음.")
    elif args.test_dir:
        npz_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.npz")))
        if not npz_paths:
            raise RuntimeError(f"'{args.test_dir}' 에 .npz 없음.")
    else:
        raise RuntimeError("--test_dir 또는 --split_json 중 하나를 지정해야 합니다.")

    test_ds = SimF81Dataset(
        npz_paths = npz_paths,
        tokenizer = tokenizer,
        pad_half  = args.pad_half,
        use_msa   = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_sim_f81,
    )
    print(f"테스트 블록 수: {len(test_ds)}")

    # ------------------------------------------------------------------
    # 3. 예측
    # ------------------------------------------------------------------
    pi_pred_list = []
    pi_true_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids  = batch["input_ids"].to(device)    # (B, Lp)
            pi_true    = batch["pi_true"]                 # (B, L, 4)  CPU
            valid_mask = batch["valid_mask"]              # (B, L)     CPU

            logits_dict = model(input_ids)
            pi_pred     = logits_dict_to_pi(logits_dict).cpu()  # (B, L_out, 4)

            # L_out → L crop
            B, L, _ = pi_true.shape
            L_out   = pi_pred.shape[1]
            if L_out != L:
                crop    = (L_out - L) // 2
                pi_pred = pi_pred[:, crop: crop + L, :]

            # valid 위치만 수집
            for b in range(B):
                mask = valid_mask[b]               # (L,) bool
                pi_pred_list.append(pi_pred[b][mask].numpy())   # (n_valid, 4)
                pi_true_list.append(pi_true[b][mask].numpy())

    pi_pred_all = np.concatenate(pi_pred_list, axis=0)  # (N_total, 4)
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
