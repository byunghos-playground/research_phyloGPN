"""
src/losses/supervised_loss.py

Naive supervised loss: 예측된 π vs ground-truth π 직접 비교.

[용도]
  F81 loss (비지도, 계통수 필요)와 비교하는 baseline 모델 훈련.
  Ground-truth π = 시뮬레이션에 사용된 실제 F81 파라미터.

[Loss 정의]
  KL(π_true || π_pred) = Σ_a π_true_a * log(π_true_a / π_pred_a)
  → 유효 사이트에 대해 평균

  KL divergence를 쓰는 이유:
    - π_true, π_pred 모두 확률 분포 (simplex)
    - KL은 두 분포의 차이를 자연스럽게 측정
    - π_true가 ground-truth이므로 KL(true||pred) 방향이 적합

[주의]
  Supervised 모델은 훈련 시 msa_codes와 tree가 불필요.
  단, 시뮬레이션된 π_true를 label로 사용하므로
  오직 simulation study 맥락에서만 의미있음
  (실제 genomic 데이터에는 π_true를 알 수 없음).
"""

import torch
import torch.nn as nn


class SupervisedPiLoss(nn.Module):
    """
    KL(π_true || π_pred) averaged over valid sites.

    Parameters
    ----------
    eps : float
        log(0) 방지용 clamp 하한.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pi_pred:    torch.Tensor,   # (B, L, 4)  모델 출력 → softmax 적용된 π
        pi_true:    torch.Tensor,   # (B, L, 4)  시뮬레이션 ground-truth π
        valid_mask: torch.Tensor,   # (B, L) bool
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pi_pred : (B, L, 4)
            모델 예측 stationary frequency (softmax 이미 적용).
        pi_true : (B, L, 4)
            시뮬레이션에 사용된 실제 π (Dirichlet 샘플).
        valid_mask : (B, L) bool
            패딩 위치 제외.

        Returns
        -------
        loss : scalar tensor
            유효 사이트 평균 KL divergence.
        """
        # log(0) 방지를 위해 clamp
        pi_true_c = pi_true.clamp(min=self.eps)
        pi_pred_c = pi_pred.clamp(min=self.eps)

        # KL(true || pred) per site: (B, L)
        kl = (pi_true_c * (pi_true_c.log() - pi_pred_c.log())).sum(dim=-1)

        # 유효 사이트만
        kl = kl * valid_mask.float()

        n_valid = valid_mask.sum()
        if n_valid == 0:
            return kl.mean()
        return kl.sum() / n_valid
