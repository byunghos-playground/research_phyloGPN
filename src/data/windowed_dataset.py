"""
src/data/windowed_dataset.py

WindowedSimF81Dataset: PhyloGPN 원본 스타일 sliding-window 데이터셋.

[현재 미사용]
  현재 시뮬레이션 설계 (청크당 π 하나, L=481) 에서는 SimF81Dataset을 직접 사용.
  각 청크가 이미 481bp이므로 sliding window 불필요.
  향후 실제 게놈 데이터 (긴 서열에서 sliding window 필요) 시 재사용 가능.

[Sliding Window 방식]
  각 블록(길이 L)의 모든 위치 center ∈ [0, L-1] 에 대해
  길이 window_size(=481)의 윈도우를 생성.

  윈도우 [center-pad_half, center+pad_half] 범위에서:
    - 범위 내 위치: 실제 ref_seq 염기 + 실제 pi_true + 실제 msa_codes
    - 범위 밖 위치: '-' 패딩 + zero pi + gap msa code (비정보)

  아이템 하나:
    input_ids  : (481,)     — 토크나이즈된 481bp 윈도우
    pi_true    : (481,4)    — 윈도우 내 per-site π (패딩 위치는 0)
    msa_codes  : (481,S)    — 윈도우 내 MSA (패딩 위치는 5=gap)
    valid_mask : (481,)     — True: 실제 사이트, False: 패딩
    center_idx : int=240    — 윈도우 내 중앙 위치 인덱스 (항상 pad_half=240)

[훈련 전략]
  - F81 loss는 center_idx 위치에만 적용 (center-site loss)
  - 이 방식으로 각 center 위치의 F81 θ를 좌우 480bp 맥락을 보고 예측
  - stride > 1 로 설정하면 훈련 데이터 수를 줄일 수 있음
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import SimF81Dataset
from src.models.tokenizer import PhyloGPNTokenizer


class WindowedSimF81Dataset(Dataset):
    """
    SimF81Dataset을 sliding window로 분해하는 래퍼.

    Parameters
    ----------
    npz_paths : list[str]
        .npz 파일 경로 목록.
    tokenizer : PhyloGPNTokenizer
    window_size : int
        윈도우 크기 (홀수여야 함, 기본값 481).
    use_msa : bool
        True면 msa_codes 포함. F81 loss 훈련 시 반드시 True.
    stride : int
        슬라이딩 보폭. 1이면 모든 center 위치 사용 (최대 데이터).
        큰 값일수록 빠른 훈련 (데이터 수 감소).
    """

    def __init__(
        self,
        npz_paths:   List[str],
        tokenizer:   PhyloGPNTokenizer,
        window_size: int  = 481,
        use_msa:     bool = True,
        stride:      int  = 1,
    ):
        super().__init__()
        assert window_size % 2 == 1, "window_size는 홀수여야 합니다 (예: 481)."

        self.window_size = window_size
        self.pad_half    = window_size // 2   # 240
        self.stride      = stride
        self.use_msa     = use_msa
        self.tokenizer   = tokenizer

        # 베이스 데이터셋 (pad 없이, 블록 단위 로드)
        self.base = SimF81Dataset(
            npz_paths = npz_paths,
            tokenizer = tokenizer,
            pad_half  = 0,
            use_msa   = use_msa,
        )

        # (block_idx, center_pos) 인덱스 테이블 생성
        # 각 블록의 모든 center 위치를 stride 간격으로 열거
        self._index: List[Tuple[int, int]] = []
        for b_idx in range(len(self.base)):
            block = self.base._load_block(b_idx)
            L     = len(block["ref_seq"])
            for center in range(0, L, stride):
                self._index.append((b_idx, center))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        block_idx, center = self._index[idx]
        block   = self.base._load_block(block_idx)

        ref_seq: str            = block["ref_seq"]
        pi_true: np.ndarray    = block["pi_true"]   # (L,4)
        L = len(ref_seq)

        # 윈도우 범위 [center - pad_half, center + pad_half]
        left  = center - self.pad_half
        right = center + self.pad_half + 1   # exclusive

        window_chars: List[str]       = []
        window_pi:    List[np.ndarray] = []
        valid_mask:   List[bool]      = []

        if self.use_msa:
            msa_codes: np.ndarray = block["msa_codes"]  # (L,S)
            S = msa_codes.shape[1]
            window_msa: List[np.ndarray] = []

        for pos in range(left, right):
            if 0 <= pos < L:
                # 실제 사이트
                window_chars.append(ref_seq[pos])
                window_pi.append(pi_true[pos])
                valid_mask.append(True)
                if self.use_msa:
                    window_msa.append(msa_codes[pos])
            else:
                # 범위 밖 → 패딩
                window_chars.append("-")
                window_pi.append(np.zeros(4, dtype=np.float32))
                valid_mask.append(False)
                if self.use_msa:
                    # 5=gap: 비정보, Felsenstein에서 all-ones로 처리됨
                    window_msa.append(np.full(S, 5, dtype=np.int64))

        window_seq = "".join(window_chars)
        window_pi_arr  = np.stack(window_pi,  axis=0)   # (W,4)
        valid_mask_arr = np.array(valid_mask, dtype=bool)

        encoded   = self.tokenizer([window_seq], return_tensors="pt",
                                   padding=False, truncation=False)
        input_ids = encoded["input_ids"][0]   # (W,)

        sample: Dict[str, Any] = {
            "input_ids":  input_ids,
            "pi_true":    torch.from_numpy(window_pi_arr),        # (W,4)
            "valid_mask": torch.from_numpy(valid_mask_arr),       # (W,)
            "center_idx": self.pad_half,    # 항상 240 (윈도우 중앙)
            "pad_id":     self.tokenizer.pad_token_id,
            "path":       self.base.npz_paths[block_idx],
        }

        if self.use_msa:
            sample["msa_codes"] = torch.from_numpy(
                np.stack(window_msa, axis=0)   # (W,S)
            )

        return sample
