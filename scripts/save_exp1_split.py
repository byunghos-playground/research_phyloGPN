"""
scripts/save_exp1_split.py

exp1_baseline 훈련이 이미 완료된 후 split.json을 사후 생성.
train_*.py와 동일한 seed=42, train_ratio=0.8, valid_ratio=0.1 사용.

Usage:
  python scripts/save_exp1_split.py
"""

import glob
import json
import os
import random

DATA_DIR    = "data/exp1_baseline/processed"
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED        = 42

OUT_DIRS = [
    "checkpoints/exp1_baseline/f81",
    "checkpoints/exp1_baseline/f81_supervised",
    "checkpoints/exp1_baseline/naive",
]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
if not paths:
    raise RuntimeError(f"'{DATA_DIR}' 에 .npz 파일이 없습니다.")

random.seed(SEED)
random.shuffle(paths)
n       = len(paths)
n_train = int(n * TRAIN_RATIO)
n_valid = int(n * VALID_RATIO)

split = {
    "train": paths[:n_train],
    "valid": paths[n_train:n_train + n_valid],
    "test":  paths[n_train + n_valid:],
}

print(f"split: train={len(split['train'])}, valid={len(split['valid'])}, test={len(split['test'])}")

for out_dir in OUT_DIRS:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "split.json")
    with open(out_path, "w") as f:
        json.dump(split, f)
    print(f"저장: {out_path}")
