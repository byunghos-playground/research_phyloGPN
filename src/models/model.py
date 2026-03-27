"""
src/models/model.py

PhyloGPN 모델 아키텍처: RCEByteNet (Reverse-Complement Equivariant ByteNet).

[논문 출처]
  Albors et al. 2025 (RECOMB) — A Phylogenetic Approach to Genomic Language Modeling
  아키텍처 원본: CARP (Yang et al. 2024, Cell Systems) → ByteNet (Kalchbrenner et al. 2017)

[핵심 특성]
  1. RCE (Reverse-Complement Equivariance):
     - DNA는 양쪽 가닥이 있어 forward/reverse complement가 동등해야 함
     - weight parametrization으로 강제: W_rc = (W + flip(W)) / 2
     - 덕분에 파라미터 수는 절반 (weight sharing)

  2. Dilated Convolution (ByteNet 블록):
     - dilation으로 수용 영역을 exponential하게 확대
     - 40 blocks: dilation = [1,5,1,5,...] (20 stack × [5^0,5^1])
     - 총 수용 영역 = 481 bp

  3. 출력:
     - 각 위치마다 4개 logit (A,C,G,T의 F81 θ 파라미터)
     - sliding window로 적용 → 전체 서열의 모든 위치에 대해 θ 예측

[입출력]
  input : (B, L_padded)  — token ID 시퀀스 (padded with '-'=5)
  output: dict {'A':(B,L_out), 'C':(B,L_out), 'G':(B,L_out), 'T':(B,L_out)}
    where L_out = L_padded - (RF - 1) due to valid convolutions (no padding in conv)
"""

from functools import cached_property
from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import PreTrainedModel

from .configuration import PhyloGPNConfig


# ---------------------------------------------------------------------------
# RCE Weight / Bias parametrizations
# ---------------------------------------------------------------------------

def _check_involution(indices: List[int]) -> bool:
    """indices가 involution (자기 자신의 역함수)인지 확인."""
    return all(indices[indices[i]] == i for i in range(len(indices)))


def _get_involution_indices(size: int) -> List[int]:
    """size 길이 reversed 인덱스 → involution (reverse-complement 대응)."""
    return list(reversed(range(size)))


class RCEWeight(nn.Module):
    """
    Conv1d weight에 RC-equivariance 강제하는 parametrization.
    W_symmetric = (W + W_flipped) / 2
    - output channel: complement 채널과 weight 공유
    - input channel: complement 채널과 weight 공유 (reversed)
    - kernel: 뒤집어서 공유
    """

    def __init__(self, input_inv: List[int], output_inv: List[int]):
        assert _check_involution(input_inv) and _check_involution(output_inv)
        super().__init__()
        self._in_inv  = input_inv
        self._out_inv = output_inv
        self._in_t    = None   # device-cached tensor
        self._out_t   = None
        self._device  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._in_t  = torch.tensor(self._in_inv,  device=x.device)
            self._out_t = torch.tensor(self._out_inv, device=x.device)
            self._device = x.device
        # x: (out, in, kernel)
        return (x + x[self._out_t][:, self._in_t].flip(2)) / 2


class IEBias(nn.Module):
    """
    Bias에 RC-equivariance 강제하는 parametrization.
    b_symmetric = (b + b_complement) / 2
    """

    def __init__(self, inv: List[int]):
        assert _check_involution(inv)
        super().__init__()
        self._inv = inv
        self._t   = None
        self._dev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._dev != x.device:
            self._t   = torch.tensor(self._inv, device=x.device)
            self._dev = x.device
        return (x + x[self._t]) / 2


class IEWeight(nn.Module):
    """Embedding weight에 RC-equivariance 강제."""

    def __init__(self, input_inv: List[int], output_inv: List[int]):
        assert _check_involution(input_inv) and _check_involution(output_inv)
        super().__init__()
        self._in_inv  = input_inv
        self._out_inv = output_inv
        self._in_t    = None
        self._out_t   = None
        self._device  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._in_t  = torch.tensor(self._in_inv,  device=x.device)
            self._out_t = torch.tensor(self._out_inv, device=x.device)
            self._device = x.device
        return (x + x[self._in_t][:, self._out_t]) / 2


# ---------------------------------------------------------------------------
# ByteNet Block (dilated residual block)
# ---------------------------------------------------------------------------

class RCEByteNetBlock(nn.Module):
    """
    단일 RCE-equivariant ByteNet residual block.

    구조 (각 conv에 RCE weight parametrization 적용):
      GroupNorm → GELU → Conv1d(outer→inner, k=1)
      GroupNorm → GELU → Conv1d(inner→inner, k=kernel, dil=dilation)  ← dilated
      GroupNorm → GELU → Conv1d(inner→outer, k=1)
      + residual (center-cropped input)

    Note: conv에 padding=0 → 출력 길이 = 입력 길이 - (k-1)*dilation
    """

    def __init__(
        self,
        outer_inv:    List[int],
        inner_dim:    int,
        kernel_size:  int,
        dilation_rate: int = 1,
    ):
        outer_dim = len(outer_inv)
        assert outer_dim % 2 == 0, "outer_dim must be even"
        assert inner_dim % 2 == 0, "inner_dim must be even"
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        super().__init__()
        inner_inv = _get_involution_indices(inner_dim)

        layers = [
            nn.GroupNorm(1, outer_dim),
            nn.GELU(),
            nn.Conv1d(outer_dim, inner_dim, kernel_size=1),          # pointwise
            nn.GroupNorm(1, inner_dim),
            nn.GELU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size, dilation=dilation_rate),  # dilated
            nn.GroupNorm(1, inner_dim),
            nn.GELU(),
            nn.Conv1d(inner_dim, outer_dim, kernel_size=1),          # pointwise
        ]

        # RCE weight parametrization 등록
        parametrize.register_parametrization(layers[2], "weight", RCEWeight(outer_inv, inner_inv))
        parametrize.register_parametrization(layers[2], "bias",   IEBias(inner_inv))
        parametrize.register_parametrization(layers[5], "weight", RCEWeight(inner_inv, inner_inv))
        parametrize.register_parametrization(layers[5], "bias",   IEBias(inner_inv))
        parametrize.register_parametrization(layers[8], "weight", RCEWeight(inner_inv, outer_inv))
        parametrize.register_parametrization(layers[8], "bias",   IEBias(outer_inv))

        self.layers       = nn.Sequential(*layers)
        self._kernel_size  = kernel_size
        self._dilation    = dilation_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, outer_dim, L_in)
        returns: (B, outer_dim, L_out)   where L_out = L_in - (k-1)*dilation
        """
        out = self.layers(x)
        # residual: center-crop x to match out length
        diff = x.shape[2] - out.shape[2]
        a    = diff // 2
        if a == 0:
            return out + x
        return out + x[:, :, a: a + out.shape[2]]


# ---------------------------------------------------------------------------
# RCEByteNet (full model backbone)
# ---------------------------------------------------------------------------

class RCEByteNet(nn.Module):
    """
    PhyloGPN 백본 네트워크.

    input_inv  = [3,2,1,0,4,5] — vocab 6개 (A,C,G,T,N,-) 의 RC involution
      A(0)↔T(3), C(1)↔G(2), N(4)↔N(4), -(5)↔-(5)
    output_inv = [3,2,1,0]    — 출력 4개 (A,C,G,T logits) 의 RC involution
    """

    def __init__(
        self,
        input_inv:  List[int],
        output_inv: List[int],
        dilation_rates: List[int],
        outer_dim:  int,
        inner_dim:  int,
        kernel_size: int,
        pad_token_idx: Optional[int] = None,
    ):
        super().__init__()
        vocab_size  = len(input_inv)
        outer_inv   = _get_involution_indices(outer_dim)
        output_dim  = len(output_inv)

        # Embedding (vocab → outer_dim), RC-equivariant
        self.embedding = nn.Embedding(vocab_size, outer_dim, padding_idx=pad_token_idx)
        parametrize.register_parametrization(
            self.embedding, "weight", IEWeight(input_inv, outer_inv)
        )
        nn.init.normal_(self.embedding.weight, std=2 ** 0.5)
        if pad_token_idx is not None:
            self.embedding.weight.data[pad_token_idx].zero_()

        # 40 ByteNet blocks (dilated residual convolutions)
        self.blocks = nn.Sequential(*[
            RCEByteNetBlock(outer_inv, inner_dim, kernel_size, r)
            for r in dilation_rates
        ])

        # 출력 projection: outer_dim → 4 (A,C,G,T logits), RC-equivariant
        self.output_layers = nn.Sequential(
            nn.GroupNorm(1, outer_dim),
            nn.GELU(),
            nn.Conv1d(outer_dim, output_dim, kernel_size=1),
        )
        parametrize.register_parametrization(
            self.output_layers[2], "weight", RCEWeight(outer_inv, output_inv)
        )
        parametrize.register_parametrization(
            self.output_layers[2], "bias",   IEBias(output_inv)
        )

        self._embedding_inv = outer_inv

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        960-dim embedding 추출 (BEND 등 downstream task 용).
        input_ids: (B, L_padded)
        returns  : (B, L_out, outer_dim)
        """
        x = self.embedding(input_ids).swapaxes(1, 2)   # (B, outer_dim, L)
        x = self.output_layers[0](self.blocks(x))       # GroupNorm 까지
        return x.swapaxes(1, 2)                         # (B, L_out, outer_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L_padded)
        returns  : (B, L_out, 4)  — [θ_A, θ_C, θ_G, θ_T] logits per position
        """
        x = self.get_embeddings(input_ids).swapaxes(1, 2)   # (B, outer_dim, L_out)
        return self.output_layers[1:](x).swapaxes(1, 2)     # (B, L_out, 4)


# ---------------------------------------------------------------------------
# PhyloGPNModel (HuggingFace wrapper)
# ---------------------------------------------------------------------------

class PhyloGPNModel(PreTrainedModel):
    """
    HuggingFace PreTrainedModel 래퍼.

    forward()는 dict {'A','C','G','T'} 형태로 반환 (기존 loss 코드 호환).
    """

    config_class = PhyloGPNConfig

    # vocab RC involution: A(0)↔T(3), C(1)↔G(2), N(4)↔N(4), -(5)↔-(5)
    _INPUT_INV  = [3, 2, 1, 0, 4, 5]
    # output RC involution: A(0)↔T(3), C(1)↔G(2)
    _OUTPUT_INV = [3, 2, 1, 0]

    def __init__(self, config: PhyloGPNConfig, **kwargs):
        super().__init__(config, **kwargs)

        # dilation 패턴: num_stacks × [k^0, k^1, ..., k^(stack_size-1)]
        dilation_rates = config.num_stacks * [
            config.kernel_size ** i for i in range(config.stack_size)
        ]

        self._model = RCEByteNet(
            input_inv      = self._INPUT_INV,
            output_inv     = self._OUTPUT_INV,
            dilation_rates = dilation_rates,
            outer_dim      = config.outer_dim,
            inner_dim      = config.inner_dim,
            kernel_size    = config.kernel_size,
            pad_token_idx  = 5,   # '-' token id
        )

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """960-dim embedding 반환. (B, L_out, 960)"""
        return self._model.get_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        input_ids : (B, L_padded)

        Returns
        -------
        dict with keys 'A','C','G','T', each (B, L_out)
            L_out = L_padded - (RF - 1) = L_padded - 480
        """
        out = self._model(input_ids)   # (B, L_out, 4)
        return {
            "A": out[:, :, 0],
            "C": out[:, :, 1],
            "G": out[:, :, 2],
            "T": out[:, :, 3],
        }
