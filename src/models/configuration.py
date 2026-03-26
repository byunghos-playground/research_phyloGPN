"""
src/models/configuration.py

PhyloGPN 모델 하이퍼파라미터 설정.

[아키텍처 개요]
  PhyloGPN = RCEByteNet (Reverse-Complement Equivariant ByteNet)
  - 40개의 residual dilated convolution block (ByteNet 스타일)
  - dilation 패턴: num_stacks=20 × [5^0, 5^1] = [1, 5, 1, 5, ..., 1, 5] (40개)
  - receptive field: 1 + Σ(kernel_size-1)*dilation = 1 + 20*(4*1 + 4*5) = 481 bp
  - 출력: 중앙 위치의 F81 파라미터 θ = (θ_A, θ_C, θ_G, θ_T) — 논문과 동일

[파라미터 설명]
  - outer_dim (960): 각 residual block의 채널 수
  - inner_dim (480): block 내부 bottleneck 채널 수 (=outer_dim/2)
  - kernel_size (5): 각 dilated conv의 커널 크기
  - stack_size (2): 한 stack당 dilation 개수 (5^0, 5^1)
  - num_stacks (20): stack 반복 횟수 → 총 40 blocks
"""

from transformers import PretrainedConfig


class PhyloGPNConfig(PretrainedConfig):
    """
    PhyloGPN 모델 설정 클래스.

    HuggingFace PretrainedConfig를 상속하므로,
    from_pretrained / save_pretrained 사용 가능.
    """

    model_type = "phylogpn"

    def __init__(
        self,
        outer_dim:  int = 960,   # residual block 채널 수 (embedding dimension)
        inner_dim:  int = 480,   # bottleneck 채널 수 (= outer_dim // 2)
        kernel_size: int = 5,    # dilated conv 커널 크기 (홀수여야 함)
        stack_size: int = 2,     # 1 stack = [5^0, 5^1, ..., 5^(stack_size-1)]
        num_stacks: int = 20,    # stack 반복 수 → 총 block 수 = num_stacks * stack_size
        **kwargs,
    ):
        self.outer_dim   = outer_dim
        self.inner_dim   = inner_dim
        self.kernel_size = kernel_size
        self.stack_size  = stack_size
        self.num_stacks  = num_stacks
        super().__init__(**kwargs)

    @property
    def n_blocks(self) -> int:
        """총 ByteNet block 수 = num_stacks * stack_size."""
        return self.num_stacks * self.stack_size

    @property
    def receptive_field(self) -> int:
        """
        모델의 수용 영역(bp) 계산.
        RF = 1 + Σ_{block} (kernel_size - 1) * dilation_rate
        """
        dilation_rates = self.num_stacks * [
            self.kernel_size ** i for i in range(self.stack_size)
        ]
        return 1 + sum((self.kernel_size - 1) * r for r in dilation_rates)
