"""
data/simulate/simulate_f81.py

pyvolve를 이용한 F81 모델 시뮬레이션 (chunk 단위).

[시뮬레이션 설계]
  목표: 각 사이트마다 다른 F81 파라미터(π)를 가진 alignment 생성.

  방식:
    1. 계통수(241-mammalian Newick) 로드
    2. L개 사이트 각각에 대해:
       - Dirichlet(1,1,1,1)에서 π = (π_A, π_C, π_G, π_T) 샘플
       - F81 모델로 pyvolve Partition(size=1) 생성
    3. 전체 partitions를 하나의 Evolver로 시뮬레이션
       → 출력: FASTA (모든 종의 시뮬레이션 서열)
    4. 각 사이트의 π를 pi.txt로 저장 (훈련 label / 검증 ground truth)

[Dirichlet(1,1,1,1) 선택 이유]
  - symmetric Dirichlet(α=1) = uniform distribution over the simplex
  - 따라서 π가 편향 없이 다양한 evolutionary constraint를 시뮬레이션
  - 극단적 π(예: π_A→1.0)도 포함 → 모델이 다양한 경우를 학습

[주의]
  pyvolve는 각 Partition을 독립적으로 진화시키므로
  L=10000 사이트에 10000개 Partition을 쓰면 매우 느릴 수 있음.
  SLURM array job으로 chunk 단위 병렬화 권장.

Usage:
  python simulate_f81.py \\
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \\
    --L 10000 \\
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
    p.add_argument("--L", type=int, default=10_000,
                   help="시뮬레이션할 사이트 수 (기본값: 10000)")
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

    # 2) 사이트별 π 샘플 + Partition 생성
    print(f"사이트 {args.L}개 시뮬레이션 준비 중 (Dirichlet α={args.alpha})...")
    partitions: list = []
    site_pis:   list = []

    for site_i in range(args.L):
        # Dirichlet(α, α, α, α) 샘플 → [π_A, π_C, π_G, π_T]
        freqs = np.random.dirichlet([args.alpha] * 4)

        model = pyvolve.Model(
            "nucleotide",
            parameters={"model": "F81", "state_freqs": freqs.tolist()},
        )
        # size=1: 이 Partition은 1개 사이트만 시뮬레이션
        partitions.append(pyvolve.Partition(models=model, size=1))
        site_pis.append(freqs)

    # 3) 시뮬레이션 실행
    aln_path = f"{args.out_prefix}.fasta"
    print(f"시뮬레이션 실행 중 → {aln_path}")
    evolver = pyvolve.Evolver(tree=tree, partitions=partitions)
    evolver(seqfile=aln_path, ratefile=None, infofile=None)

    # 4) π 저장
    pi_path = f"{args.out_prefix}_pi.txt"
    with open(pi_path, "w") as f:
        f.write("#site\tpi_A\tpi_C\tpi_G\tpi_T\n")
        for i, freqs in enumerate(site_pis, start=1):
            f.write(f"{i}\t{freqs[0]:.8f}\t{freqs[1]:.8f}\t"
                    f"{freqs[2]:.8f}\t{freqs[3]:.8f}\n")

    print(f"완료: {aln_path}, {pi_path}")
    print(f"  → 사이트 수: {args.L}, π shape: ({args.L}, 4)")


if __name__ == "__main__":
    main()
