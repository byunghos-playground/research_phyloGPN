"""
data/simulate/simulate_exp3_gc.py

[Exp3: gc] GC OU process + F81 시뮬레이션 (rate variation 없음).

[설계]
  - GC(x) ~ Ornstein-Uhlenbeck(μ=0.41, σ=0.07, θ=1/5000)
    → correlation length = 5,000 bp
    → clip to [0.30, 0.65]
  - π(x) = [(1-GC(x))/2, GC(x)/2, GC(x)/2, (1-GC(x))/2]  # [A, C, G, T]
  - rate = 1.0 (constant, no per-site variation)
  - Genome length: L=10,000 bp

[exp1/exp2와의 차이]
  - π가 chunk 전체에서 하나가 아닌 position마다 다름
  - ref_seq frequency counting → 481bp window 평균 π만 얻음 → shortcut 차단
  - pi.txt에 position마다 다른 π 저장 (L rows, each row different)

[출력]
  - {prefix}.fasta        : 전체 species 서열 (L=10000 bp)
  - {prefix}_pi.txt       : position별 π (L rows)

Usage:
  python simulate_exp3_gc.py \\
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \\
    --L 10000 \\
    --out_prefix data/exp3_gc/raw/genome_00000 \\
    --seed 0
"""

import argparse
import numpy as np
import ete3

_INT_TO_CHAR = ['A', 'C', 'G', 'T']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GC OU process + F81 시뮬레이션 (no rate variation)."
    )
    p.add_argument("--tree_path",  type=str, required=True)
    p.add_argument("--L",          type=int, default=10000)
    p.add_argument("--out_prefix", type=str, required=True)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--mu",         type=float, default=0.41,   help="OU mean GC")
    p.add_argument("--sigma",      type=float, default=0.07,   help="OU stationary std")
    p.add_argument("--theta",      type=float, default=1/5000, help="OU mean-reversion rate")
    p.add_argument("--gc_min",     type=float, default=0.30)
    p.add_argument("--gc_max",     type=float, default=0.65)
    return p.parse_args()


def simulate_gc_ou(L: int, mu: float, sigma: float, theta: float,
                   gc_min: float, gc_max: float) -> np.ndarray:
    """
    GC content를 OU process로 L position에 걸쳐 시뮬레이션.

    exact discrete update:
      GC[i] = GC[i-1] * exp(-θ) + μ * (1-exp(-θ)) + N(0, σ² * (1-exp(-2θ)))
    """
    exp_theta  = np.exp(-theta)
    noise_std  = sigma * np.sqrt(1.0 - np.exp(-2.0 * theta))

    gc = np.empty(L)
    gc[0] = np.clip(np.random.normal(mu, sigma), gc_min, gc_max)
    for i in range(1, L):
        gc[i] = gc[i-1] * exp_theta + mu * (1.0 - exp_theta) + np.random.normal(0.0, noise_std)
    return np.clip(gc, gc_min, gc_max)


def gc_to_pi(gc: np.ndarray) -> np.ndarray:
    """
    GC(x) → π(x) = [(1-GC)/2, GC/2, GC/2, (1-GC)/2]  [A, C, G, T]
    Returns (L, 4) float32.
    """
    at = (1.0 - gc) / 2.0
    cg = gc / 2.0
    return np.stack([at, cg, cg, at], axis=1).astype(np.float32)


def f81_forward_simulate(tree: ete3.Tree,
                         pi_per_site: np.ndarray) -> dict:
    """
    Position-varying π로 F81 forward simulation.

    Parameters
    ----------
    pi_per_site : (L, 4) float — per-site stationary frequencies
    rates       : None (rate=1 everywhere)

    Returns
    -------
    dict {leaf_name: (L,) int array}
    """
    L = len(pi_per_site)
    node_states: dict = {}

    for node in tree.traverse("preorder"):
        nid = id(node)

        if node.is_root():
            cum = np.cumsum(pi_per_site, axis=1)     # (L, 4)
            u   = np.random.random(L)
            node_states[nid] = (cum < u[:, None]).sum(axis=1).clip(0, 3).astype(np.int64)

        else:
            parent_states = node_states[id(node.up)]
            t = float(node.dist)

            exp_t = np.exp(-t)                                       # scalar (rate=1)
            probs = pi_per_site * (1.0 - exp_t)                      # (L, 4)
            probs[np.arange(L), parent_states] += exp_t

            cum    = np.cumsum(probs, axis=1)
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

    print(f"트리 로드 중: {args.tree_path}")
    tree = ete3.Tree(args.tree_path, format=1)
    leaf_order = [n.name for n in tree.traverse() if n.is_leaf()]
    print(f"  → leaf 수: {len(leaf_order)}")

    # 1) GC OU process
    gc = simulate_gc_ou(args.L, args.mu, args.sigma, args.theta,
                        args.gc_min, args.gc_max)
    print(f"GC: mean={gc.mean():.4f}, min={gc.min():.4f}, max={gc.max():.4f}")

    # 2) π per site
    pi_per_site = gc_to_pi(gc)   # (L, 4)

    # 3) Forward simulation
    print(f"시뮬레이션 실행 중 ({args.L} 사이트, {len(leaf_order)} 종)...")
    leaf_seqs = f81_forward_simulate(tree, pi_per_site)

    # 4) FASTA 저장
    aln_path = f"{args.out_prefix}.fasta"
    write_fasta(leaf_seqs, leaf_order, aln_path)
    print(f"FASTA 저장: {aln_path}")

    # 5) pi.txt 저장 (position별 다른 π)
    pi_path = f"{args.out_prefix}_pi.txt"
    with open(pi_path, 'w') as f:
        f.write("#site\tpi_A\tpi_C\tpi_G\tpi_T\n")
        for i, pi in enumerate(pi_per_site, 1):
            f.write(f"{i}\t{pi[0]:.8f}\t{pi[1]:.8f}\t{pi[2]:.8f}\t{pi[3]:.8f}\n")
    print(f"pi.txt 저장: {pi_path}")

    print("완료.")


if __name__ == "__main__":
    main()
