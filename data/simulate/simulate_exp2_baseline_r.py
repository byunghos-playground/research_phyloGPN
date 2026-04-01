"""
data/simulate/simulate_exp2_baseline_r.py

[Exp2: baseline_r] F81 + per-site rate variation 시뮬레이션.

[baseline (exp1)과의 차이]
  - 청크당 π 하나: 동일 (Dirichlet(1,1,1,1))
  - per-site r ~ Gamma(shape=0.5): 각 사이트마다 독립적인 rate multiplier
  - 시뮬레이션: pyvolve 대신 custom F81 forward simulation (ete3 + numpy)
    → per-site r_true를 완벽히 추적 가능
  - 출력: chunk.fasta + chunk_pi.txt (exp1 호환) + chunk_r.txt (r_true 추가)

[F81 전이 확률]
  P(j | i, t) = π_j + (δ_ij - π_j) × exp(-r × t)
  where t = branch_length, r = per-site rate

[설계 의도]
  - ref_seq frequency로 π 추정은 여전히 가능하나 r=낮은 site에서 noisy
  - MSA column: r=높으면 equilibrium(≈π), r=낮으면 ancestral state 반영
  - → shortcut이 완전히 막히지는 않지만 더 어려워짐
  - r_true를 저장하므로 "high-r vs low-r site별 π 예측 정확도" 사후 분석 가능

Usage:
  python simulate_exp2_baseline_r.py \\
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \\
    --L 481 \\
    --out_prefix data/exp2_baseline_r/raw/chunk_000 \\
    --seed 42
"""

import argparse
import numpy as np
import ete3


# 정수 코드 → 뉴클레오타이드 문자
_INT_TO_CHAR = ['A', 'C', 'G', 'T']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F81 + per-site rate variation 시뮬레이션 (custom forward simulation)."
    )
    p.add_argument("--tree_path",  type=str, required=True,
                   help="Newick 계통수 파일 경로")
    p.add_argument("--L",          type=int, default=481,
                   help="시뮬레이션할 사이트 수 (기본값: 481)")
    p.add_argument("--out_prefix", type=str, required=True,
                   help="출력 prefix. {prefix}.fasta, {prefix}_pi.txt, {prefix}_r.txt 생성")
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--alpha",      type=float, default=1.0,
                   help="Dirichlet α (π 샘플링, 기본값: 1.0)")
    p.add_argument("--rate_shape", type=float, default=0.5,
                   help="Gamma shape for per-site rate (기본값: 0.5)")
    return p.parse_args()


def f81_forward_simulate(
    tree:   ete3.Tree,
    pi:     np.ndarray,   # (4,) float
    rates:  np.ndarray,   # (L,) float, per-site rate multipliers
) -> dict:
    """
    F81 모델로 계통수 위에서 forward (top-down) 시뮬레이션.

    Parameters
    ----------
    tree  : ete3.Tree
    pi    : (4,) stationary frequencies [A, C, G, T]
    rates : (L,) per-site rate multipliers (r ~ Gamma)

    Returns
    -------
    dict {leaf_name: np.ndarray (L,) int}
        각 leaf의 per-site 뉴클레오타이드 코드 (A=0, C=1, G=2, T=3)
    """
    L = len(rates)
    node_states: dict = {}  # id(node) → (L,) int array

    for node in tree.traverse("preorder"):
        nid = id(node)

        if node.is_root():
            # Root: sample from π (same π for all sites)
            node_states[nid] = np.random.choice(4, size=L, p=pi)

        else:
            parent_states = node_states[id(node.up)]   # (L,) int
            t = float(node.dist)

            # F81 전이 확률 (vectorized over L sites):
            # P(j | i, r, t) = π_j + (δ_ij - π_j) × exp(-r × t)
            #                 = π_j × (1 - exp(-r×t))  [j ≠ i]
            #                 + exp(-r×t)               [j = i]
            exp_rt = np.exp(-rates * t)                    # (L,)
            probs  = np.outer(1.0 - exp_rt, pi)            # (L, 4)
            probs[np.arange(L), parent_states] += exp_rt   # diagonal +exp(-r×t)

            # Vectorized categorical sampling
            cum    = np.cumsum(probs, axis=1)              # (L, 4)
            u      = np.random.random(L)
            states = (cum < u[:, None]).sum(axis=1).clip(0, 3)

            node_states[nid] = states.astype(np.int64)

    return {
        node.name: node_states[id(node)]
        for node in tree.traverse()
        if node.is_leaf()
    }


def write_fasta(leaf_seqs: dict, leaf_order: list, out_path: str) -> None:
    with open(out_path, 'w') as f:
        for name in leaf_order:
            seq = ''.join(_INT_TO_CHAR[s] for s in leaf_seqs[name])
            f.write(f">{name}\n{seq}\n")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # 1) 계통수 로드
    print(f"트리 로드 중: {args.tree_path}")
    tree = ete3.Tree(args.tree_path, format=1)
    leaf_order = [n.name for n in tree.traverse() if n.is_leaf()]
    print(f"  → leaf 수: {len(leaf_order)}")

    # 2) 청크 전체 π 하나 샘플 (baseline과 동일)
    pi = np.random.dirichlet([args.alpha] * 4)
    print(f"π = A:{pi[0]:.4f} C:{pi[1]:.4f} G:{pi[2]:.4f} T:{pi[3]:.4f}")

    # 3) per-site rate r ~ Gamma(shape, scale=1.0)
    rates = np.random.gamma(shape=args.rate_shape, scale=1.0, size=args.L)
    print(f"r ~ Gamma({args.rate_shape}): "
          f"mean={rates.mean():.4f}, min={rates.min():.4f}, max={rates.max():.4f}")

    # 4) Forward simulation
    print(f"시뮬레이션 실행 중 ({args.L} 사이트, {len(leaf_order)} 종)...")
    leaf_seqs = f81_forward_simulate(tree, pi, rates)

    # 5) FASTA 저장 (fasta_to_npz.py 호환 포맷)
    aln_path = f"{args.out_prefix}.fasta"
    write_fasta(leaf_seqs, leaf_order, aln_path)
    print(f"FASTA 저장: {aln_path}")

    # 6) pi.txt 저장 (모든 행 동일 π — fasta_to_npz.py 호환)
    pi_path = f"{args.out_prefix}_pi.txt"
    with open(pi_path, 'w') as f:
        f.write("#site\tpi_A\tpi_C\tpi_G\tpi_T\n")
        for i in range(1, args.L + 1):
            f.write(f"{i}\t{pi[0]:.8f}\t{pi[1]:.8f}\t{pi[2]:.8f}\t{pi[3]:.8f}\n")
    print(f"pi.txt 저장: {pi_path}")

    # 7) r.txt 저장 (per-site rate — exp2 고유)
    r_path = f"{args.out_prefix}_r.txt"
    with open(r_path, 'w') as f:
        f.write("#site\tr\n")
        for i, r in enumerate(rates, 1):
            f.write(f"{i}\t{r:.8f}\n")
    print(f"r.txt 저장: {r_path}")

    print("완료.")


if __name__ == "__main__":
    main()
