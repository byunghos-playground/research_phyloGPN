"""
data/simulate/split_data.py

processed/*.npz 파일들을 train/valid/test 세트로 분할.

[분할 전략]
  - 블록 단위로 분할 (사이트 단위 아님)
  - 기본 비율: train 80%, valid 10%, test 10%
  - 랜덤 셔플 후 분할 (seed 고정으로 재현성 보장)
  - 파일을 복사하지 않고 symlink 생성 (디스크 절약)

Usage:
  python split_data.py \\
    --processed_dir data/processed \\
    --train_dir     data/train \\
    --valid_dir     data/valid \\
    --test_dir      data/test \\
    --train_ratio   0.8 \\
    --valid_ratio   0.1 \\
    --seed          42
"""

import argparse
import glob
import os
import random
import shutil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="npz 블록 train/valid/test 분할.")
    p.add_argument("--processed_dir", type=str, default="data/processed",
                   help="변환된 .npz 파일이 있는 디렉토리")
    p.add_argument("--train_dir",     type=str, default="data/train")
    p.add_argument("--valid_dir",     type=str, default="data/valid")
    p.add_argument("--test_dir",      type=str, default="data/test")
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--valid_ratio",   type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--copy",          action="store_true",
                   help="symlink 대신 파일 복사 (NFS에서 symlink 문제 시 사용)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # test_ratio = 나머지
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    assert test_ratio > 0, "train + valid 비율 합이 1.0 이하여야 합니다."

    # .npz 파일 목록
    npz_files = sorted(glob.glob(os.path.join(args.processed_dir, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"'{args.processed_dir}' 에 .npz 파일이 없습니다.")
    print(f"총 {len(npz_files)}개 블록 발견.")

    # 셔플
    random.seed(args.seed)
    random.shuffle(npz_files)

    # 분할 인덱스
    n       = len(npz_files)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    n_test  = n - n_train - n_valid

    splits = {
        args.train_dir: npz_files[:n_train],
        args.valid_dir: npz_files[n_train: n_train + n_valid],
        args.test_dir:  npz_files[n_train + n_valid:],
    }

    print(f"분할: train={n_train}, valid={n_valid}, test={n_test}")

    for dest_dir, files in splits.items():
        os.makedirs(dest_dir, exist_ok=True)
        for src in files:
            fname = os.path.basename(src)
            dst   = os.path.join(dest_dir, fname)

            # 이미 존재하면 skip
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)

            if args.copy:
                shutil.copy2(src, dst)
            else:
                # 절대경로 symlink (NFS 환경에서도 안전)
                os.symlink(os.path.abspath(src), dst)

        action = "복사" if args.copy else "symlink"
        print(f"  {dest_dir}: {len(files)}개 {action} 완료")

    print("분할 완료.")


if __name__ == "__main__":
    main()
