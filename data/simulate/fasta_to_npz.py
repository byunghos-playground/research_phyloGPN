"""
data/simulate/fasta_to_npz.py

FASTA + pi.txt → .npz 변환 스크립트.

[입력]
  - FASTA 파일: pyvolve 시뮬레이션 출력 (모든 종의 서열, 같은 길이 L)
  - pi.txt 파일: 각 사이트의 실제 F81 파라미터 (simulate_f81.py 출력)

[출력 .npz 키]
  ref_seq    : str (길이 L) — 첫 번째 종(또는 --ref_name 지정 종)의 서열
               → 모델 입력 x^(i) 역할 (논문의 "reference genome")
  pi_true    : (L, 4) float32 — 실제 stationary frequency
               → supervised 훈련의 label, 검증 시 ground truth
  msa_codes  : (L, S) int64 — 전체 alignment 정수 코드
               → F81 loss 계산에 필요 (y^(i) in 논문)
  taxon_names: (S,) object — 종 이름 배열
               → tree leaf order와 동일 순서여야 함

[인코딩 규칙]
  A→0, C→1, G→2, T→3, N(ambiguous)→4, -/기타→5

Usage:
  python fasta_to_npz.py \\
    --fasta data/raw/chunk_000.fasta \\
    --pi    data/raw/chunk_000_pi.txt \\
    --out   data/processed/block_000.npz \\
    --ref_name Homo_sapiens   # 없으면 첫 번째 종 사용
"""

import argparse
import numpy as np
from typing import List, Tuple, Optional


# 뉴클레오타이드 문자 → 정수 코드
_CHAR_TO_CODE = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
    "N": 4, "n": 4,
    "-": 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FASTA + pi.txt → .npz 변환."
    )
    p.add_argument("--fasta",    type=str, required=True,  help="입력 FASTA 파일")
    p.add_argument("--pi",       type=str, required=True,  help="입력 pi.txt 파일")
    p.add_argument("--out",      type=str, required=True,  help="출력 .npz 파일 경로")
    p.add_argument("--ref_name", type=str, default=None,
                   help="ref_seq로 사용할 종 이름 (없으면 첫 번째 종)")
    return p.parse_args()


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """FASTA 파일 읽기 → (taxon_names, sequences)."""
    names: List[str] = []
    seqs:  List[str] = []
    name:  Optional[str] = None
    buf:   List[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(buf))
                name = line[1:].split()[0]  # 첫 단어만 (설명 제거)
                names.append(name)
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs.append("".join(buf))

    if not names:
        raise RuntimeError(f"FASTA {path}에 서열이 없습니다.")

    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise RuntimeError(f"서열 길이가 일정하지 않습니다: {lengths}")

    return names, seqs


def read_pi(path: str, expected_L: int) -> np.ndarray:
    """
    pi.txt → (L, 4) float32 배열.

    파일 형식:
      #site  pi_A  pi_C  pi_G  pi_T
      1      0.25  0.25  0.25  0.25
      ...
    """
    pi_list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            pi_list.append([float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])])

    if len(pi_list) != expected_L:
        raise RuntimeError(
            f"pi 행 수({len(pi_list)}) ≠ 서열 길이({expected_L})"
        )
    return np.asarray(pi_list, dtype=np.float32)


def encode_msa(seqs: List[str]) -> np.ndarray:
    """
    (S,) 문자열 리스트 → (L, S) int64 배열.
    A=0, C=1, G=2, T=3, N=4, -=5, 기타=4(N으로 처리).
    """
    S = len(seqs)
    L = len(seqs[0])
    msa = np.zeros((L, S), dtype=np.int64)
    for s_idx, seq in enumerate(seqs):
        for i, ch in enumerate(seq):
            msa[i, s_idx] = _CHAR_TO_CODE.get(ch, 4)  # 미지 문자 → N
    return msa


def main() -> None:
    args = parse_args()

    # 1) FASTA 읽기
    taxon_names, seqs = read_fasta(args.fasta)
    L = len(seqs[0])
    S = len(seqs)
    print(f"FASTA 로드: {args.fasta}  (S={S} 종, L={L} 사이트)")

    # 2) π 읽기
    pi_true = read_pi(args.pi, expected_L=L)
    print(f"pi 로드: {args.pi}  shape={pi_true.shape}")

    # 3) MSA 인코딩 (L, S)
    msa_codes = encode_msa(seqs)
    print(f"MSA 인코딩 완료: shape={msa_codes.shape}")

    # 4) ref_seq 선택
    if args.ref_name is None:
        ref_idx  = 0
        ref_name = taxon_names[0]
    else:
        if args.ref_name not in taxon_names:
            raise RuntimeError(f"'{args.ref_name}' 이 FASTA에 없습니다.")
        ref_idx  = taxon_names.index(args.ref_name)
        ref_name = args.ref_name
    ref_seq = seqs[ref_idx]
    print(f"ref_seq: '{ref_name}' (index {ref_idx})")

    # 5) .npz 저장
    taxon_arr = np.array(taxon_names, dtype=object)
    np.savez_compressed(
        args.out,
        ref_seq     = ref_seq,
        pi_true     = pi_true,
        msa_codes   = msa_codes,
        taxon_names = taxon_arr,
    )
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
