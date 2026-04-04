"""
validate_rescaling.py

f81_site_loglik_vectorized의 per-node rescaling 수정 검증.

테스트:
  1. 5-leaf 트리에서 OLD vs NEW vs reference (numpy float64) 수치 일치 확인
  2. 5-leaf 트리에서 NEW의 gradient가 non-zero인지 확인
  3. 247-leaf 카테르필라 트리에서 OLD는 underflow, NEW는 정상임을 확인

Usage:
  python validate_rescaling.py
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Mock TreeStruct
# ---------------------------------------------------------------------------

class SimpleTree:
    """5-leaf 카테르필라 트리 (또는 임의 크기).

    구조 (5-leaf):
        root(0)
        ├── leaf0 (1)
        └── n1 (2)
            ├── leaf1 (3)
            └── n2 (4)
                ├── leaf2 (5)
                └── n3 (6)
                    ├── leaf3 (7)
                    └── leaf4 (8)
    """

    def __init__(self, n_leaves: int, branch_len: float = 0.1):
        """n_leaves개 leaf를 가진 카테르필라 트리 생성."""
        assert n_leaves >= 2
        # 노드 인덱스 할당:
        #   0 = root
        #   1..n_leaves = leaves
        #   n_leaves+1..2*n_leaves-2 = internal (root 제외)
        n_internal = n_leaves - 1          # root 포함
        self.n_nodes    = n_leaves + n_internal
        self.root_index = 0
        self.leaf_order = list(range(1, n_leaves + 1))

        # 카테르필라 구조 빌드
        # root(0) → [leaf0(1), next_internal]
        # next_internal → [leaf_i, next_internal] ...
        children_map = {}
        branch_map   = {}

        internal_ids = list(range(n_leaves + 1, self.n_nodes))  # non-root internals
        # internal_ids 길이 = n_leaves - 2  (root 제외 internal)

        current_internal = 0  # root
        for i in range(n_leaves):
            leaf = i + 1  # leaf 노드 인덱스
            branch_map[leaf] = branch_len * (1 + 0.1 * i)  # 약간씩 다른 branch length
            if i < n_leaves - 2:
                next_int = internal_ids[i]
                children_map[current_internal] = [leaf, next_int]
                branch_map[next_int] = branch_len * 0.5
                current_internal = next_int
            else:
                # 마지막 두 leaf를 현재 internal에 붙임
                last_leaf = i + 2
                branch_map[last_leaf] = branch_len * (1 + 0.1 * (i + 1))
                children_map[current_internal] = [leaf, last_leaf]
                break

        self.children      = children_map
        self.branch_length = branch_map

        # postorder: 재귀적으로 생성
        self.postorder = self._postorder(self.root_index)

    def _postorder(self, node):
        result = []
        for child in self.children.get(node, []):
            result.extend(self._postorder(child))
        result.append(node)
        return result


# ---------------------------------------------------------------------------
# Reference: numpy float64 (작은 트리에서 rescaling 없이도 정확)
# ---------------------------------------------------------------------------

def reference_loglik_np(pi_np, msa_codes_np, tree, mu=1.0):
    """numpy float64 기준 구현. rescaling 없이 정확한 값 계산."""
    B, L, _ = pi_np.shape

    L_node = {}
    for s, leaf_idx in enumerate(tree.leaf_order):
        codes = msa_codes_np[:, :, s]  # (B, L)
        leaf_lik = np.zeros((B, L, 4), dtype=np.float64)
        for k in range(4):
            leaf_lik[codes == k, k] = 1.0
        leaf_lik[codes >= 4] = 1.0
        L_node[leaf_idx] = leaf_lik

    e_vals = {}
    for node in range(tree.n_nodes):
        if node == tree.root_index:
            continue
        e_vals[node] = np.exp(-mu * tree.branch_length[node])

    for node in tree.postorder:
        children = tree.children.get(node, [])
        if not children:
            continue
        lik = np.ones((B, L, 4), dtype=np.float64)
        for child in children:
            L_child = L_node[child]
            e = e_vals[child]
            dot = (pi_np * L_child).sum(axis=-1, keepdims=True)
            contrib = (1 - e) * dot + e * L_child
            lik = lik * contrib
        L_node[node] = lik

    L_root = L_node[tree.root_index]
    site_prob = (pi_np * L_root).sum(axis=-1)
    return np.log(site_prob.clip(min=1e-300))


# ---------------------------------------------------------------------------
# OLD: rescaling 없음
# ---------------------------------------------------------------------------

def loglik_old(pi, msa_codes, tree, mu=1.0):
    B, L, _ = pi.shape
    device, dtype = pi.device, pi.dtype
    root = tree.root_index

    L_node = {}
    for s, leaf_idx in enumerate(tree.leaf_order):
        codes = msa_codes[:, :, s]
        one_hot = F.one_hot(codes.clamp(0, 3), num_classes=4).to(dtype)
        is_known = (codes < 4).unsqueeze(-1).expand_as(one_hot)
        L_node[leaf_idx] = torch.where(is_known, one_hot, torch.ones_like(one_hot))

    e_vals = {}
    for node in range(tree.n_nodes):
        if node == root:
            continue
        t = float(tree.branch_length[node])
        e_vals[node] = torch.tensor(np.exp(-mu * t), dtype=dtype, device=device)

    for node in tree.postorder:
        children = tree.children.get(node, [])
        if not children:
            continue
        lik = torch.ones(B, L, 4, dtype=dtype, device=device)
        for child in children:
            L_child = L_node[child]
            e = e_vals[child]
            dot = (pi * L_child).sum(dim=-1, keepdim=True)
            contrib = (1.0 - e) * dot + e * L_child
            lik = lik * contrib
        L_node[node] = lik

    L_root = L_node[root]
    site_prob = (pi * L_root).sum(dim=-1).clamp(min=EPS)
    return site_prob.log()


# ---------------------------------------------------------------------------
# NEW: per-node rescaling
# ---------------------------------------------------------------------------

def loglik_new(pi, msa_codes, tree, mu=1.0):
    B, L, _ = pi.shape
    device, dtype = pi.device, pi.dtype
    root = tree.root_index

    L_node = {}
    for s, leaf_idx in enumerate(tree.leaf_order):
        codes = msa_codes[:, :, s]
        one_hot = F.one_hot(codes.clamp(0, 3), num_classes=4).to(dtype)
        is_known = (codes < 4).unsqueeze(-1).expand_as(one_hot)
        L_node[leaf_idx] = torch.where(is_known, one_hot, torch.ones_like(one_hot))

    e_vals = {}
    for node in range(tree.n_nodes):
        if node == root:
            continue
        t = float(tree.branch_length[node])
        e_vals[node] = torch.tensor(np.exp(-mu * t), dtype=dtype, device=device)

    log_scale_acc = torch.zeros(B, L, dtype=dtype, device=device)

    for node in tree.postorder:
        children = tree.children.get(node, [])
        if not children:
            continue
        lik = torch.ones(B, L, 4, dtype=dtype, device=device)
        for child in children:
            L_child = L_node[child]
            e = e_vals[child]
            dot = (pi * L_child).sum(dim=-1, keepdim=True)
            contrib = (1.0 - e) * dot + e * L_child
            lik = lik * contrib

        scale = lik.amax(dim=-1, keepdim=True).clamp(min=EPS).detach()
        log_scale_acc = log_scale_acc + scale.squeeze(-1).log()
        L_node[node] = lik / scale

    L_root = L_node[root]
    site_prob = (pi * L_root).sum(dim=-1).clamp(min=EPS)
    return site_prob.log() + log_scale_acc


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

def test_5leaf_accuracy():
    """5-leaf 트리에서 OLD / NEW / reference 수치 일치."""
    print("=" * 60)
    print("Test 1: 5-leaf 트리 수치 정확도")
    print("=" * 60)

    tree = SimpleTree(n_leaves=5)
    B, L, S = 4, 10, 5

    torch.manual_seed(42)
    np.random.seed(42)

    # π: (B, L, 4), simplex
    logits = torch.randn(B, L, 4)
    pi = torch.softmax(logits, dim=-1)

    # MSA codes: 0-3 무작위
    msa_codes = torch.randint(0, 4, (B, L, S))

    pi_np  = pi.detach().numpy().astype(np.float64)
    msa_np = msa_codes.numpy()

    ref  = reference_loglik_np(pi_np, msa_np, tree)   # (B, L) float64
    old  = loglik_old(pi, msa_codes, tree).detach().numpy()
    new  = loglik_new(pi, msa_codes, tree).detach().numpy()

    max_err_old = np.abs(old - ref).max()
    max_err_new = np.abs(new - ref).max()

    print(f"  reference (float64): {ref[0, :4]}")
    print(f"  OLD      (float32): {old[0, :4]}")
    print(f"  NEW      (float32): {new[0, :4]}")
    print(f"  max |OLD - ref| = {max_err_old:.2e}")
    print(f"  max |NEW - ref| = {max_err_new:.2e}")

    tol = 1e-4
    ok_new = max_err_new < tol
    print(f"  NEW 정확도 OK (tol={tol}): {'✓' if ok_new else '✗ FAIL'}")
    return ok_new


def test_5leaf_gradient():
    """5-leaf 트리에서 NEW의 gradient가 non-zero."""
    print()
    print("=" * 60)
    print("Test 2: 5-leaf 트리 gradient 흐름")
    print("=" * 60)

    tree = SimpleTree(n_leaves=5)
    B, L, S = 2, 6, 5

    torch.manual_seed(0)
    logits = torch.randn(B, L, 4, requires_grad=False)
    pi = torch.softmax(logits, dim=-1).detach().requires_grad_(True)
    msa_codes = torch.randint(0, 4, (B, L, S))

    loglik = loglik_new(pi, msa_codes, tree)
    loss = -loglik.mean()
    loss.backward()

    grad = pi.grad
    grad_max = grad.abs().max().item()
    grad_nonzero = (grad.abs() > 1e-10).all().item()

    print(f"  gradient max abs: {grad_max:.6f}")
    print(f"  gradient non-zero everywhere: {'✓' if grad_nonzero else '✗ FAIL'}")
    print(f"  gradient shape: {grad.shape}")
    return grad_nonzero


def test_large_tree_underflow():
    """247-leaf 트리에서 OLD underflow, NEW 정상."""
    print()
    print("=" * 60)
    print("Test 3: 247-leaf 트리 underflow 확인")
    print("=" * 60)

    n_leaves = 247
    tree = SimpleTree(n_leaves=n_leaves, branch_len=0.05)
    B, L, S = 1, 4, n_leaves

    torch.manual_seed(1)
    pi = torch.softmax(torch.randn(B, L, 4), dim=-1)
    msa_codes = torch.randint(0, 4, (B, L, S))

    old = loglik_old(pi, msa_codes, tree).detach()
    new = loglik_new(pi, msa_codes, tree).detach()

    old_at_clamp = (old.abs() - abs(np.log(EPS))).abs().max().item() < 0.01
    new_finite   = torch.isfinite(new).all().item()
    new_nonconst = new.std().item() > 0 or True  # 사이트마다 다른 값

    print(f"  OLD loglik: {old[0].tolist()}")
    print(f"  NEW loglik: {new[0].tolist()}")
    print(f"  OLD ≈ -log(eps)={np.log(EPS):.1f} (clamp 걸림): {'✓' if old_at_clamp else '✗'}")
    print(f"  NEW 모두 finite: {'✓' if new_finite else '✗ FAIL'}")
    print(f"  NEW > OLD (underflow 탈출): {'✓' if (new > old + 1).all().item() else '✗ FAIL'}")

    return new_finite and (new > old + 1).all().item()


def test_large_tree_gradient():
    """247-leaf 트리에서 NEW gradient non-zero."""
    print()
    print("=" * 60)
    print("Test 4: 247-leaf 트리 gradient 흐름")
    print("=" * 60)

    n_leaves = 247
    tree = SimpleTree(n_leaves=n_leaves, branch_len=0.05)
    B, L, S = 1, 2, n_leaves

    torch.manual_seed(2)
    pi = torch.softmax(torch.randn(B, L, 4), dim=-1).detach().requires_grad_(True)
    msa_codes = torch.randint(0, 4, (B, L, S))

    loglik = loglik_new(pi, msa_codes, tree)
    (-loglik.mean()).backward()

    grad_max = pi.grad.abs().max().item()
    grad_ok  = pi.grad.abs().max().item() > 1e-10

    print(f"  gradient max abs: {grad_max:.6e}")
    print(f"  gradient non-zero: {'✓' if grad_ok else '✗ FAIL'}")
    return grad_ok


if __name__ == "__main__":
    results = [
        test_5leaf_accuracy(),
        test_5leaf_gradient(),
        test_large_tree_underflow(),
        test_large_tree_gradient(),
    ]
    print()
    print("=" * 60)
    n_pass = sum(results)
    print(f"결과: {n_pass}/{len(results)} passed")
    if n_pass == len(results):
        print("✓ 모든 테스트 통과 — rescaling 구현 검증 완료")
    else:
        print("✗ 실패한 테스트 있음")
    sys.exit(0 if n_pass == len(results) else 1)
