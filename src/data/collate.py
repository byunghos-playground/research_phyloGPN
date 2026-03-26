"""
src/data/collate.py

DataLoader collate 함수 모음.

[두 가지 collate]
  1. collate_sim_f81:
     SimF81Dataset 전용 (블록 단위, 가변 길이 L).
     배치 내 최대 길이로 패딩.

  2. collate_windowed_sim_f81:
     WindowedSimF81Dataset 전용 (고정 길이 W=481).
     모든 아이템 길이가 같으므로 단순 stack.
"""

from typing import Any, Dict, List

import torch


def collate_sim_f81(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SimF81Dataset 배치 collate.

    가변 길이 L을 배치 내 최대값으로 패딩:
      - input_ids: pad_id('-'=5) 로 패딩
      - pi_true:   0으로 패딩
      - msa_codes: -1로 패딩 (유효하지 않은 사이트 표시)

    Returns dict keys:
      input_ids   : (B, max_Lp)   — 패딩된 토큰 ID
      pi_true     : (B, max_L, 4) — 패딩된 stationary frequencies
      valid_mask  : (B, max_L)    — True: 실제 사이트
      length      : (B,)          — 각 블록의 실제 길이
      msa_codes   : (B, max_L, S) — 있을 때만 포함
      paths       : list[str]
    """
    B      = len(batch)
    pad_id = batch[0]["pad_id"]

    # input_ids 패딩 (Lp = L + 2*pad_half, 배치마다 L이 다를 수 있음)
    max_Lp    = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((B, max_Lp), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        lp = item["input_ids"].size(0)
        input_ids[i, :lp] = item["input_ids"]

    # pi_true, valid_mask 패딩
    lengths = [int(item["length"]) for item in batch]
    max_L   = max(lengths)
    pi_true    = torch.zeros(B, max_L, 4, dtype=torch.float32)
    valid_mask = torch.zeros(B, max_L, dtype=torch.bool)
    for i, item in enumerate(batch):
        L = item["length"]
        pi_true[i, :L, :]  = item["pi_true"]
        valid_mask[i, :L]  = True

    result: Dict[str, Any] = {
        "input_ids":  input_ids,
        "pi_true":    pi_true,
        "valid_mask": valid_mask,
        "length":     torch.tensor(lengths, dtype=torch.long),
        "paths":      [item["path"] for item in batch],
    }

    # msa_codes (선택)
    if "msa_codes" in batch[0]:
        S         = batch[0]["msa_codes"].size(1)
        msa_codes = torch.full((B, max_L, S), fill_value=-1, dtype=torch.long)
        for i, item in enumerate(batch):
            L = item["length"]
            msa_codes[i, :L, :] = item["msa_codes"]
        result["msa_codes"] = msa_codes

    return result


def collate_windowed_sim_f81(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    WindowedSimF81Dataset 배치 collate.

    모든 아이템이 같은 window_size(W=481)이므로 단순 stack.

    Returns dict keys:
      input_ids   : (B, W)    — 윈도우 토큰 ID
      pi_true     : (B, W, 4) — 윈도우 내 π (패딩 위치 0)
      valid_mask  : (B, W)    — True: 실제 사이트
      center_idx  : (B,)      — 중앙 위치 인덱스 (항상 240)
      msa_codes   : (B, W, S) — 있을 때만 포함
      paths       : list[str]
    """
    input_ids  = torch.stack([item["input_ids"]  for item in batch])   # (B,W)
    pi_true    = torch.stack([item["pi_true"]    for item in batch])   # (B,W,4)
    valid_mask = torch.stack([item["valid_mask"] for item in batch])   # (B,W)
    center_idx = torch.tensor(
        [item["center_idx"] for item in batch], dtype=torch.long
    )  # (B,)

    result: Dict[str, Any] = {
        "input_ids":  input_ids,
        "pi_true":    pi_true,
        "valid_mask": valid_mask,
        "center_idx": center_idx,
        "paths":      [item["path"] for item in batch],
    }

    if "msa_codes" in batch[0]:
        result["msa_codes"] = torch.stack(
            [item["msa_codes"] for item in batch]   # (B,W,S)
        )

    return result
