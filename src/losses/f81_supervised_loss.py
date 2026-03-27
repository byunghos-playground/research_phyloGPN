"""
src/losses/f81_supervised_loss.py

F81 Supervised loss (f81_supervised 모델용).

[위치]
  F81 (unsupervised) 과 Naive (직접 supervised) 사이의 중간 방식.
  Physics-Informed ML 프레임워크에서 π_true를 간접적으로 활용.

[아이디어]
  F81 loss는 π_pred로 alignment likelihood를 최대화.
  하지만 π_true가 있다면, "π_true로 계산한 likelihood가 π_pred로 계산한
  likelihood보다 높으면 → π_pred가 틀린 것" 이라는 신호를 쓸 수 있음.

[Loss 정의]
  log_lik_true = log P_F81(alignment | π_true, T)   ← oracle likelihood
  log_lik_pred = log P_F81(alignment | π_pred, T)   ← 모델 예측 likelihood

  loss = log_lik_true - log_lik_pred
       = log (P_true / P_pred)

  이를 minimize하면 P_pred → P_true, 즉 모델이 π_true가 내는 것과 같은
  alignment likelihood를 내도록 학습됨.

  π를 직접 비교하는 게 아니라, likelihood를 통해 간접적으로 π_true 정보를 씀.

[F81 loss와의 차이]
  F81:            loss = -log P_F81(alignment | π_pred, T)
  f81_supervised: loss = log P_F81(alignment | π_true, T)
                       - log P_F81(alignment | π_pred, T)

  f81_supervised는 π_true로 계산한 likelihood가 고정 target이 됨.
  (gradient는 π_pred 부분에서만 흐름 — π_true는 상수로 취급)
"""

from typing import Dict

import torch
import torch.nn as nn

from src.utils.math_f81 import logits_dict_to_pi, f81_site_loglik_vectorized


class F81SupervisedLoss(nn.Module):
    """
    F81 likelihood matching loss.

    loss = log P_F81(alignment | π_true, T) - log P_F81(alignment | π_pred, T)

    Parameters
    ----------
    tree_struct : TreeStruct
        tree_utils.py의 TreeStruct 인스턴스.
    mu : float
        F81 rate scaler (기본값 1.0).
    eps : float
        log(0) 방지용 clamp 하한.
    """

    def __init__(self, tree_struct, mu: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.tree = tree_struct
        self.mu   = mu
        self.eps  = eps

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],  # {'A','C','G','T'}: (B, L_out)
        msa_codes:   torch.Tensor,             # (B, L, S)
        pi_true:     torch.Tensor,             # (B, L, 4)
        valid_mask:  torch.Tensor,             # (B, L) bool
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits_dict : dict
            모델 forward() 출력. 각 value shape = (B, L_out).
        msa_codes : (B, L, S)
            MSA alignment columns.
        pi_true : (B, L, 4)
            시뮬레이션 ground-truth π. gradient 흐르지 않음 (detach).
        valid_mask : (B, L) bool

        Returns
        -------
        loss : scalar tensor
            유효 사이트 평균 (log P_true - log P_pred).
        """
        B, L, S = msa_codes.shape

        # 1) logits → π_pred (B, L_out, 4)
        pi_pred = logits_dict_to_pi(logits_dict)
        L_out   = pi_pred.shape[1]

        # 2) L_out ≠ L 이면 crop
        if L_out != L:
            if L_out < L or (L_out - L) % 2 != 0:
                raise ValueError(
                    f"[F81SupervisedLoss] 길이 불일치: pi L_out={L_out} vs msa L={L}."
                )
            crop    = (L_out - L) // 2
            pi_pred = pi_pred[:, crop: crop + L, :]   # (B, L, 4)

        # 3) log P_F81(alignment | π_pred, T)  — gradient 흐름
        loglik_pred = f81_site_loglik_vectorized(
            pi         = pi_pred,
            msa_codes  = msa_codes,
            tree       = self.tree,
            mu         = self.mu,
            valid_mask = valid_mask,
            eps        = self.eps,
        )

        # 4) log P_F81(alignment | π_true, T)  — gradient 없음 (상수 target)
        pi_true_d = pi_true.detach().clamp(min=self.eps)
        # softmax 보장: pi_true는 이미 simplex이지만 clamp 후 renormalize
        pi_true_d = pi_true_d / pi_true_d.sum(dim=-1, keepdim=True)

        loglik_true = f81_site_loglik_vectorized(
            pi         = pi_true_d,
            msa_codes  = msa_codes,
            tree       = self.tree,
            mu         = self.mu,
            valid_mask = valid_mask,
            eps        = self.eps,
        )

        # 5) loss = log P_true - log P_pred  (유효 사이트 평균)
        #    minimize → P_pred → P_true
        loss_per_site = (loglik_true - loglik_pred) * valid_mask.float()
        n_valid       = valid_mask.sum()
        if n_valid == 0:
            return loss_per_site.mean()
        return loss_per_site.sum() / n_valid
