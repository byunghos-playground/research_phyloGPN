"""
src/losses/f81_loss.py

F81 phylogenetic likelihood loss (PhyloGPN 훈련용).

[배경]
  PhyloGPN 논문(Albors et al. 2025)에서 F81 loss를 사용하지만
  구체적인 구현 코드는 공개되지 않음. 이 파일은 논문의 수식을 바탕으로
  직접 구현한 것임.

[Loss 정의 — 논문 Eq.(4)]
  L(W; D) = -(1/n) Σ_i log P_F81(y^(i) | f_W(x^(i)), T^(i))
           + (1/n) Σ_i log π^(ref,i)(f_W)

  여기서:
    f_W(x^(i)) = θ (F81 파라미터)
    π^(i)      = softmax(θ) = stationary frequency
    P_F81      = Felsenstein pruning으로 계산한 alignment column likelihood
    두 번째 항 = conditioning term

[conditioning term의 역할]
  reference genome x는 모델 input이면서 MSA y의 leaf 0에도 포함됨.
  Felsenstein pruning에서 ref 위치 염기(예: A)는 one-hot leaf likelihood로
  처리되므로, 모델이 ref 염기에 π mass를 집중하면 P_F81이 인위적으로 높아짐.

  conditioning term "+ (1/n) Σ log π_ref" 를 loss에 더함으로써
  이 shortcut을 penalize:
    - minimize L 방향에서 log π_ref가 크면(π_ref → 1) 불리 → ref 염기 편향 억제
    - 결과적으로 모델은 다른 종들의 alignment에서 진화적 신호를 학습

  ref species는 msa_codes의 index 0 (fasta_to_npz.py의 default ref_idx=0 대응).

[버그 수정]
  구버전 f81_site_loglik_batch():
    - root를 postorder loop에서 skip → L_node[root] = None → crash
    - Python B×L 이중 for loop → 매우 느림
  수정:
    - math_f81.py의 f81_site_loglik_vectorized() 사용 (PyTorch 벡터화)
"""

from typing import Dict

import torch
import torch.nn as nn

from src.utils.math_f81 import logits_dict_to_pi, f81_site_loglik_vectorized


class F81LikelihoodLoss(nn.Module):
    """
    F81 phylogenetic NLL loss.

    훈련 시에는 중앙 위치(center site)에만 loss를 계산함.
    (→ train_f81.py에서 valid_mask_center 로 제어)

    Parameters
    ----------
    tree_struct : TreeStruct
        tree_utils.py의 TreeStruct 인스턴스.
        msa_codes의 S 축 순서와 tree.leaf_order가 대응되어야 함.
    mu : float
        F81 rate scaler (기본값 1.0).
    eps : float
        log(0) 방지용 clamp 하한.
    """

    def __init__(self, tree_struct, mu: float = 1.0, eps: float = 1e-12,
                 use_conditioning: bool = True):
        super().__init__()
        self.tree              = tree_struct
        self.mu                = mu
        self.eps               = eps
        self.use_conditioning  = use_conditioning

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],  # {'A','C','G','T'}: (B, L_out)
        msa_codes:   torch.Tensor,             # (B, L, S)
        valid_mask:  torch.Tensor,             # (B, L) bool
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits_dict : dict
            모델 forward() 출력. 각 value shape = (B, L_out).
            L_out은 conv의 valid padding으로 L보다 작을 수 있음.
        msa_codes : (B, L, S)
            MSA alignment columns. S = 종 수 = tree.n_leaves.
        valid_mask : (B, L) bool
            True인 위치만 loss에 포함.
            훈련 시 center-only: valid_mask_center[b, center_idx[b]] = True만.

        Returns
        -------
        loss : scalar tensor
            유효 사이트에 대한 평균 NLL.
        """
        B, L, S = msa_codes.shape

        # 1) logits → π  (B, L_out, 4)
        pi = logits_dict_to_pi(logits_dict)
        L_out = pi.shape[1]

        # 2) L_out ≠ L 인 경우: 중앙 L개만 슬라이싱
        #    (conv valid padding으로 양쪽이 동일하게 줄어들므로 대칭 crop)
        if L_out != L:
            if L_out < L or (L_out - L) % 2 != 0:
                raise ValueError(
                    f"[F81Loss] 길이 불일치: pi L_out={L_out} vs msa L={L}. "
                    "conv 패딩 설정을 확인하세요."
                )
            crop = (L_out - L) // 2
            pi = pi[:, crop: crop + L, :]   # (B, L, 4)

        # 3) Felsenstein pruning → per-site log-likelihood (B, L)
        loglik = f81_site_loglik_vectorized(
            pi         = pi,
            msa_codes  = msa_codes,
            tree       = self.tree,
            mu         = self.mu,
            valid_mask = valid_mask,
            eps        = self.eps,
        )

        # 4) Conditioning term: + (1/n) Σ log π^(ref)  [논문 Eq.(4) 두 번째 항]
        #    ref species = msa_codes[:, :, 0] (fasta_to_npz.py default: ref_idx=0)
        ref_codes  = msa_codes[:, :, 0]                            # (B, L), 0-5
        known_ref  = (ref_codes < 4)                               # (B, L) bool — N/gap 제외
        safe_codes = ref_codes.clamp(0, 3)                         # gather 안전 인덱스
        pi_ref = torch.gather(
            pi, dim=-1, index=safe_codes.unsqueeze(-1)
        ).squeeze(-1)                                              # (B, L)
        # valid이고 ref 염기가 known인 위치에만 conditioning 적용
        log_pi_ref = torch.log(pi_ref.clamp(min=self.eps)) * (valid_mask & known_ref).float()

        # 5) 총 loss = NLL (+ conditioning), 유효 사이트 평균
        nll = -loglik * valid_mask.float()
        if self.use_conditioning:
            loss_per_site = nll + log_pi_ref
        else:
            loss_per_site = nll
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return loss_per_site.mean()
        return loss_per_site.sum() / n_valid
