"""
src/utils/math_f81.py

F81 모델에 필요한 수학 함수 모음.

[F81 모델 요약]
  - 각 사이트마다 stationary frequency π = (π_A, π_C, π_G, π_T) 가 있음
  - 전이 확률: P_ij(t) = π_j + exp(-μt) * (δ_ij - π_j)
    → i → j 로 시간 t 동안 evolve할 확률
  - 이 모델의 핵심: off-diagonal은 π_j에만 의존하고 i와 무관함
    → P @ L_child 를 O(4^2) 대신 O(4) 로 계산 가능 (vectorization 핵심)

[Felsenstein Pruning 요약]
  - Leaf: observed nucleotide에 one-hot (N/-: all-ones, 비정보적)
  - Internal node (postorder): L_node[k] = Π_children contrib_k
    contrib_k = Σ_j P_kj * L_child[j]
               = (1 - e^{-μt}) * (π · L_child) + e^{-μt} * L_child[k]
  - Site likelihood: Σ_k π_k * L_root[k]

[버그 수정 내역]
  - 구버전: for loop 내에서 `if node == root: continue` 로 root를 skip해서
    L_node[root]가 영원히 None → 훈련 시 즉시 ValueError 발생.
    → 수정: root도 내부 노드와 동일하게 처리 (P_edge[root]는 사용하지 않음)
  - 구버전: B × L Python 이중 for loop → 매우 느림
    → 수정: 모든 (B, L) 쌍을 동시에 처리하는 vectorized 구현으로 교체
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. logits → stationary probability π
# ---------------------------------------------------------------------------

def logits_dict_to_pi(logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    PhyloGPN 모델 출력(logits dict)을 per-site F81 stationary frequency π 로 변환.

    Parameters
    ----------
    logits_dict : dict[str, Tensor]
        키: 'A', 'C', 'G', 'T', 각 값의 shape = (B, L)

    Returns
    -------
    pi : (B, L, 4)   마지막 축 = [π_A, π_C, π_G, π_T]
    """
    # (B, L) 네 개를 쌓아서 (B, 4, L) 만들고, base 축(dim=1)으로 softmax
    logits = torch.stack(
        [logits_dict["A"], logits_dict["C"], logits_dict["G"], logits_dict["T"]],
        dim=1,  # (B, 4, L)
    )
    pi = torch.softmax(logits, dim=1)       # (B, 4, L)
    pi = pi.permute(0, 2, 1).contiguous()   # (B, L, 4)
    return pi


# ---------------------------------------------------------------------------
# 2. Vectorized Felsenstein pruning (핵심 수학)
# ---------------------------------------------------------------------------

def f81_site_loglik_vectorized(
    pi: torch.Tensor,           # (B, L, 4)
    msa_codes: torch.Tensor,    # (B, L, S)  — 0=A,1=C,2=G,3=T,4=N,5='-'
    tree,                       # TreeStruct (tree_utils.py)
    mu: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,  # (B, L) bool
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    F81 + Felsenstein pruning으로 per-site log-likelihood 계산.

    구버전 대비 개선사항:
      1) Python B×L 이중 루프 제거 → 모든 사이트를 텐서 연산으로 동시 처리
      2) F81 고유 성질 활용:
             contrib_k = (1 - e^{-μt}) * (π · L_child) + e^{-μt} * L_child[k]
         → 행렬 곱 없이 dot product + scalar 연산만으로 O(4) 계산

    Parameters
    ----------
    pi : (B, L, 4)
        모델이 예측한 각 사이트의 stationary distribution.
    msa_codes : (B, L, S)
        정수 코드. 0=A, 1=C, 2=G, 3=T, 4=N(unknown), 5='-'(gap).
        S는 leaf 수(tree.leaf_order 길이)와 같아야 함.
    tree : TreeStruct
        tree_utils.py 의 TreeStruct 인스턴스.
    mu : float
        전체 rate scaler (기본값 1.0).
    valid_mask : (B, L) bool or None
        패딩 위치는 False. None이면 모두 True.
    eps : float
        log(0) 방지용.

    Returns
    -------
    loglik : (B, L)
        각 사이트의 log P(alignment column | π, T).
        valid_mask=False 위치는 0.
    """
    B, L, _ = pi.shape
    S        = msa_codes.shape[2]
    device   = pi.device
    dtype    = pi.dtype
    n_nodes  = tree.n_nodes
    root     = tree.root_index

    # -----------------------------------------------------------------------
    # Step 1. Leaf likelihood 초기화
    # -----------------------------------------------------------------------
    # dict 사용: pre-allocated tensor + inplace write는 autograd version 충돌을 일으킴
    # (L_node[:, :, child, :]가 pi computation graph에 연결된 채로
    #  L_node[:, :, node, :] = lik 로 inplace 수정 → backward 시 version mismatch)
    L_node: Dict[int, torch.Tensor] = {}

    for s, leaf_idx in enumerate(tree.leaf_order):
        codes = msa_codes[:, :, s]          # (B, L), int

        # one-hot for A/C/G/T (code 0-3); all-ones for N/gap (code 4,5)
        codes_clamped = codes.clamp(0, 3)   # (B, L)
        one_hot = F.one_hot(codes_clamped, num_classes=4).to(dtype)  # (B, L, 4)

        is_known = (codes < 4).unsqueeze(-1).expand_as(one_hot)  # (B, L, 4) bool
        # known → one-hot, unknown → 1.0 (모든 상태 equally possible)
        L_node[leaf_idx] = torch.where(is_known, one_hot, torch.ones_like(one_hot))

    # -----------------------------------------------------------------------
    # Step 2. 각 branch의 exp(-μt) 미리 계산 (root 제외)
    # -----------------------------------------------------------------------
    # e_vals[node] = exp(-mu * branch_length[node])  (scalar tensor)
    e_vals: Dict[int, torch.Tensor] = {}
    for node in range(n_nodes):
        if node == root:
            continue
        t = float(tree.branch_length[node])
        e_vals[node] = torch.tensor(
            torch.exp(torch.tensor(-mu * t)).item(),
            dtype=dtype, device=device
        )

    # -----------------------------------------------------------------------
    # Step 3. Felsenstein pruning (postorder: leaves → root)
    # -----------------------------------------------------------------------
    # [수정] root를 skip하지 않음 — root도 내부 노드와 같은 방식으로 처리.
    # root는 P_edge[root]를 사용하지 않으므로 안전.
    for node in tree.postorder:
        children = tree.children[node]
        if not children:
            continue  # leaf: 이미 초기화됨

        # 내부 노드(root 포함): 자식들의 contribution을 곱함
        lik = torch.ones(B, L, 4, dtype=dtype, device=device)  # (B, L, 4)

        for child in children:
            L_child = L_node[child]             # (B, L, 4)
            e       = e_vals[child]             # scalar

            # F81 전이의 핵심 공식:
            #   contrib_k = (1-e) * (π · L_child) + e * L_child[k]
            dot     = (pi * L_child).sum(dim=-1, keepdim=True)  # (B, L, 1)
            contrib = (1.0 - e) * dot + e * L_child             # (B, L, 4)
            lik     = lik * contrib

        L_node[node] = lik

    # -----------------------------------------------------------------------
    # Step 4. Root에서 site probability 계산
    # -----------------------------------------------------------------------
    L_root   = L_node[root]                     # (B, L, 4)
    site_prob = (pi * L_root).sum(dim=-1)       # (B, L)
    site_prob = site_prob.clamp(min=eps)
    loglik    = torch.log(site_prob)            # (B, L)

    # valid_mask=False 위치는 0으로
    if valid_mask is not None:
        loglik = loglik * valid_mask.float()

    return loglik
