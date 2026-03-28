"""
src/models/tokenizer.py

PhyloGPN DNA 시퀀스 토크나이저.

[vocab]
  A=0, C=1, G=2, T=3, N=4(unknown), -(pad/gap)=5

[버그 수정]
  - 구버전 _convert_id_to_token(): self._vocab (char→int) 로 int 인덱싱 →
    dict는 str 키이므로 int로 조회하면 KeyError 발생.
    → 수정: _id_to_token (int→char) 역방향 dict 별도 생성.

[길이 제약]
  - 구버전 assert len(seq) >= 481 이 있었으나 windowed_dataset에서
    정확히 481짜리 window를 넘기므로 이 assert는 유지.
    단, SimF81Dataset 에서는 L+2*pad_half 길이 시퀀스를 넘기므로
    항상 >= 481 이 보장됨.
"""

from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer


class PhyloGPNTokenizer(PreTrainedTokenizer):
    """
    A/C/G/T/N/- 6종 문자를 정수 ID로 변환하는 DNA 토크나이저.

    vocab:
      A → 0
      C → 1
      G → 2
      T → 3
      N → 4  (ambiguous / unknown)
      - → 5  (padding / gap)

    special tokens:
      pad_token = '-' (id=5)
      unk_token = 'N' (id=4)
      나머지 special tokens (bos, eos, sep, cls, mask) = None
    """

    model_input_names = ["input_ids"]

    # 문자 → ID
    _CHAR_TO_ID: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "-": 5}
    # ID → 문자 (역방향)  [버그 수정: 구버전은 이 역방향 dict가 없었음]
    _ID_TO_CHAR: Dict[int, str] = {v: k for k, v in _CHAR_TO_ID.items()}

    def __init__(
        self,
        model_max_length: int = 10 ** 9,
        unk_token: str = "N",
        pad_token: str = "-",
        **kwargs,
    ):
        # special token 인자를 명시적으로 None으로 설정
        super().__init__(
            model_max_length=model_max_length,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=None,
            eos_token=None,
            sep_token=None,
            cls_token=None,
            mask_token=None,
            **kwargs,
        )

    def _tokenize(self, seq: str) -> List[str]:
        """
        문자열 시퀀스를 문자 리스트로 분리.

        [주의] transformers는 pad_token('-') 등 special token으로 텍스트를 먼저 분리한 뒤
        각 조각에 _tokenize를 호출한다. 따라서 이 메서드는 481보다 짧은 조각을 받을 수 있음.
        길이 체크는 _tokenize 내부에서 하면 안 됨.
        """
        return list(seq)

    def _convert_token_to_id(self, token: str) -> int:
        """문자 → ID. vocab에 없는 문자는 N(4)으로 처리."""
        return self._CHAR_TO_ID.get(token, self._CHAR_TO_ID["N"])

    def _convert_id_to_token(self, idx: int) -> str:
        """
        ID → 문자.
        [버그 수정] 구버전: self._vocab[idx] → _vocab은 {str: int}이므로
        int로 조회하면 KeyError. 수정: _ID_TO_CHAR (int→str) 사용.
        """
        return self._ID_TO_CHAR.get(idx, "N")

    @property
    def vocab_size(self) -> int:
        return len(self._CHAR_TO_ID)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._CHAR_TO_ID)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple:
        # vocab이 코드에 하드코딩되어 있으므로 별도 저장 불필요
        return ()
