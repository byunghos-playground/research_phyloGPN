"""
src/data/dataset.py

SimF81Dataset: 시뮬레이션된 F81 데이터 로더.

[데이터 포맷 (.npz)]
  각 .npz 파일 = 하나의 "block" (연속된 서열 L개 사이트)
  필수 키:
    ref_seq    : str, 길이 L — 레퍼런스(첫 번째 종) 뉴클레오타이드 서열
    pi_true    : (L, 4) float32 — 각 사이트의 실제 F81 stationary frequency
    msa_codes  : (L, S) int64  — alignment 정수 코드 (0=A,1=C,2=G,3=T,4=N,5='-')
    taxon_names: (S,) object   — 종 이름 배열 (tree leaf order와 동일 순서)

[패딩 방식]
  모델 수용 영역 RF=481 이므로 ref_seq 양쪽에 pad_half=240 개의 '-'를 붙임.
  패딩된 시퀀스 길이 = L + 2*240 = L + 480.
  모델 출력 길이 = (L + 480) - 480 = L (valid convolution).

[용도]
  - NN_train_supervised.py: use_msa=False (pi_true만 필요)
  - NN_train_phylo_windowed.py 의 WindowedSimF81Dataset 내부 사용
"""

import os
import glob
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.tokenizer import PhyloGPNTokenizer


class SimF81Dataset(Dataset):
    """
    F81 시뮬레이션 .npz 블록 데이터셋.

    한 아이템 = 하나의 npz 블록 전체 (길이 L 서열).
    모델 입력용으로 양쪽 pad_half개 '-' 패딩 추가.

    Parameters
    ----------
    npz_paths : list[str]
        .npz 파일 경로 목록.
    tokenizer : PhyloGPNTokenizer, optional
        None이면 기본 인스턴스 생성.
    pad_half : int
        양쪽 패딩 길이 (기본값 240 = RF//2 = 481//2).
    use_msa : bool
        True면 msa_codes도 반환 (F81 loss 훈련 시 필요).
        False면 pi_true만 반환 (supervised 훈련 시).
    """

    def __init__(
        self,
        npz_paths: List[str],
        tokenizer: Optional[PhyloGPNTokenizer] = None,
        pad_half:  int  = 240,      # 481 // 2
        use_msa:   bool = False,
    ):
        super().__init__()

        # 디렉토리/glob 인자도 허용
        expanded: List[str] = []
        for p in npz_paths:
            if os.path.isdir(p):
                expanded.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
            else:
                expanded.append(p)
        if not expanded:
            raise ValueError("SimF81Dataset: .npz 파일을 찾을 수 없습니다.")
        self.npz_paths = expanded

        self.tokenizer = tokenizer or PhyloGPNTokenizer(model_max_length=10 ** 9)
        self.pad_half  = pad_half
        self.use_msa   = use_msa

        # 전체 블록을 메모리에 미리 로드 (디스크 I/O 병목 제거)
        print(f"블록 캐시 로딩 중... ({len(self.npz_paths)}개 파일)")
        self._cache: List[Dict[str, Any]] = [
            self._load_block(i) for i in range(len(self.npz_paths))
        ]
        print("캐시 완료.")

    def __len__(self) -> int:
        return len(self.npz_paths)

    def _load_block(self, idx: int) -> Dict[str, Any]:
        """블록 데이터 반환. 캐시가 있으면 캐시에서, 없으면 디스크에서 로드."""
        if hasattr(self, "_cache"):
            return self._cache[idx]
        path = self.npz_paths[idx]
        data = np.load(path, allow_pickle=True)

        if "ref_seq" not in data or "pi_true" not in data:
            raise ValueError(f"{path}: 'ref_seq', 'pi_true' 키가 필요합니다.")

        block: Dict[str, Any] = {
            "ref_seq": str(data["ref_seq"]),
            "pi_true": np.asarray(data["pi_true"], dtype=np.float32),  # (L,4)
        }
        if self.use_msa:
            if "msa_codes" not in data:
                raise ValueError(f"{path}: use_msa=True이지만 'msa_codes' 없음.")
            block["msa_codes"]   = np.asarray(data["msa_codes"],   dtype=np.int64)   # (L,S)
            block["taxon_names"] = list(map(str, data["taxon_names"]))

        return block

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        block   = self._load_block(idx)
        ref_seq = block["ref_seq"]
        pi_true = block["pi_true"]      # (L,4)
        L       = len(ref_seq)

        if pi_true.shape[0] != L:
            raise ValueError(
                f"{self.npz_paths[idx]}: ref_seq 길이({L}) ≠ pi_true 행 수({pi_true.shape[0]})"
            )

        # 양쪽 '-' 패딩 추가 → 모델 입력 길이 = L + 2*pad_half
        pad_str    = "-" * self.pad_half
        padded_seq = pad_str + ref_seq + pad_str

        encoded   = self.tokenizer([padded_seq], return_tensors="pt",
                                   padding=False, truncation=False)
        input_ids = encoded["input_ids"][0]   # (L + 2*pad_half,)

        sample: Dict[str, Any] = {
            "input_ids": input_ids,                        # (Lp,)
            "pi_true":   torch.from_numpy(pi_true),       # (L,4)
            "length":    L,
            "pad_id":    self.tokenizer.pad_token_id,
            "path":      self.npz_paths[idx],
        }

        if self.use_msa:
            sample["msa_codes"]   = torch.from_numpy(block["msa_codes"])  # (L,S)
            sample["taxon_names"] = block["taxon_names"]

        return sample
