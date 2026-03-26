"""
src/utils/checkpoint.py

모델 체크포인트 저장/로드 유틸리티.

[용도]
  - 훈련 중 epoch 단위로 체크포인트 저장
  - 최적 모델(best val loss) 자동 추적
  - 훈련 재개(resume) 지원
"""

import os
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    path:       str,
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    loss:       float,
    config:     Optional[Dict[str, Any]] = None,
) -> None:
    """
    체크포인트를 파일로 저장.

    Parameters
    ----------
    path : str
        저장 경로 (예: 'checkpoints/epoch_05.pt')
    model : nn.Module
    optimizer : Optimizer
    epoch : int
        현재 epoch 번호 (1-indexed)
    loss : float
        현재 epoch의 평균 loss
    config : dict, optional
        모델 config dict (재로드 시 아키텍처 재현용)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch":                epoch,
        "loss":                 loss,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config":               config,
    }
    torch.save(ckpt, path)
    print(f"[checkpoint] saved → {path}  (epoch={epoch}, loss={loss:.6f})")


def load_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    str = "cpu",
) -> Dict[str, Any]:
    """
    체크포인트 로드 후 model / optimizer 에 state 복원.

    Parameters
    ----------
    path : str
        체크포인트 파일 경로
    model : nn.Module
        state_dict 를 복원할 모델 인스턴스
    optimizer : Optimizer, optional
        None 이면 optimizer state 는 무시 (inference 전용)
    device : str
        'cuda' 또는 'cpu'

    Returns
    -------
    dict with keys: epoch, loss, config
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"[checkpoint] loaded ← {path}  (epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 'N/A'):.6f})")
    return {
        "epoch":  ckpt.get("epoch", 0),
        "loss":   ckpt.get("loss", float("inf")),
        "config": ckpt.get("config", None),
    }


class BestModelTracker:
    """
    Validation loss 기준으로 best 모델을 자동 추적하고 저장.

    Example
    -------
    >>> tracker = BestModelTracker("checkpoints/best.pt")
    >>> for epoch in range(n_epochs):
    ...     val_loss = validate(model)
    ...     tracker.update(val_loss, model, optimizer, epoch)
    """

    def __init__(self, best_path: str):
        """
        Parameters
        ----------
        best_path : str
            best 체크포인트 저장 경로
        """
        self.best_path = best_path
        self.best_loss = float("inf")

    def update(
        self,
        val_loss:  float,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        config:    Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        val_loss 가 이전 best 보다 낮으면 체크포인트 저장.

        Returns
        -------
        bool
            True if new best, False otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(self.best_path, model, optimizer, epoch, val_loss, config)
            print(f"[checkpoint] new best! loss={val_loss:.6f}")
            return True
        return False
