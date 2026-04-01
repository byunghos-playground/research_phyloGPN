"""
data/simulate/simulate_f81.py

pyvolve를 이용한 F81 모델 시뮬레이션 (chunk 단위).

[시뮬레이션 설계]
  목표: 청크당 하나의 F81 파라미터(π)를 가진 alignment 생성.

  방식:
    1. 계통수(241-mammalian Newick) 로드
    2. Dirichlet(α,α,α,α)에서 π = (π_A, π_C, π_G, π_T) 하나를 샘플
    3. 그 π로 L개 사이트를 모두 시뮬레이션 (단일 Partition, size=L)
       → 출력: FASTA (모든 종의 시뮬레이션 서열)
    4. π를 pi.txt로 저장 (L개 행 모두 동일 값, fasta_to_npz.py 호환)

[설계 의도]
  모델은 L개 ref_seq 문맥을 보고 π를 예측. L개 사이트 전체가 같은 π에서
  생성되므로 ref_seq의 empirical frequency가 π 정보를 담음.
  → 모델이 context → π 매핑을 실제로 학습할 수 있음.

  L=481 (= RF) 권장: 각 청크가 padding 없이 모델 수용 영역과 동일한 크기.
  청크 수를 늘려 규모 조정 (SLURM array job 권장).

[Dirichlet(1,1,1,1) 선택 이유]
  - symmetric Dirichlet(α=1) = uniform distribution over the simplex
  - π가 편향 없이 다양한 경우를 커버 (극단적 π 포함)

Usage:
  python simulate_f81.py \\
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \\
    --L 481 \\
    --out_prefix data/raw/chunk_000 \\
    --seed 42
"""

import argparse
import numpy as np
import pyvolve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F81 시뮬레이션: 각 사이트마다 독립적인 π로 시뮬레이션."
    )
    p.add_argument("--tree_path", type=str, required=True,
                   help="Newick 계통수 파일 경로 (예: data/trees/241-mammalian-2020v2.1.nh.txt)")
    p.add_argument("--L", type=int, default=481,
                   help="시뮬레이션할 사이트 수 (기본값: 481 = 모델 RF)")
    p.add_argument("--out_prefix", type=str, required=True,
                   help="출력 파일 prefix. {prefix}.fasta, {prefix}_pi.txt 생성.")
    p.add_argument("--seed", type=int, default=None,
                   help="랜덤 시드 (재현성 보장용)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Dirichlet 농도 파라미터 (기본값: 1.0 = uniform over simplex)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 재현성
    if args.seed is not None:
        np.random.seed(args.seed)

    # 1) 계통수 로드
    print(f"트리 로드 중: {args.tree_path}")
    tree = pyvolve.read_tree(file=args.tree_path)

    # 2) 청크 전체에 사용할 π 하나 샘플
    print(f"사이트 {args.L}개 시뮬레이션 준비 중 (Dirichlet α={args.alpha}, 청크당 π 1개)...")
    freqs = np.random.dirichlet([args.alpha] * 4)
    print(f"  π = A:{freqs[0]:.4f}  C:{freqs[1]:.4f}  G:{freqs[2]:.4f}  T:{freqs[3]:.4f}")

    model     = pyvolve.Model(
        "nucleotide",
        parameters={"model": "F81", "state_freqs": freqs.tolist()},
    )
    partition = pyvolve.Partition(models=model, size=args.L)

    # 3) 시뮬레이션 실행
    aln_path = f"{args.out_prefix}.fasta"
    print(f"시뮬레이션 실행 중 → {aln_path}")
    evolver = pyvolve.Evolver(tree=tree, partitions=[partition])
    evolver(seqfile=aln_path, ratefile=None, infofile=None)

    # 4) π 저장 (L개 행 모두 동일 — fasta_to_npz.py 포맷 호환)
    pi_path = f"{args.out_prefix}_pi.txt"
    with open(pi_path, "w") as f:
        f.write("#site\tpi_A\tpi_C\tpi_G\tpi_T\n")
        for i in range(1, args.L + 1):
            f.write(f"{i}\t{freqs[0]:.8f}\t{freqs[1]:.8f}\t"
                    f"{freqs[2]:.8f}\t{freqs[3]:.8f}\n")

    print(f"완료: {aln_path}, {pi_path}")
    print(f"  → 사이트 수: {args.L}, π shape: ({args.L}, 4) [모두 동일 값]")


if __name__ == "__main__":
    main()
