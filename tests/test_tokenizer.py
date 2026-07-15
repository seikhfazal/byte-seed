from __future__ import annotations

import pytest

from byteseed.generate import marker_id, stop_token_ids
from byteseed.tokenizer import ByteSeedTokenizer
import byteseed.tokenizer as tokenizer_module


class FakeSentencePieceProcessor:
    pieces = {"<unk>": 0, "<s>": 1, "</s>": 2, "<|end|>": 4, "<|user|>": 5}

    def __init__(self, model_file: str):
        self.model_file = model_file
        self.decoded_ids: list[int] = []

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def unk_id(self) -> int:
        return 0

    def get_piece_size(self) -> int:
        return 10

    def encode(self, text: str, out_type):
        assert out_type is int
        return [6, 7]

    def decode(self, ids: list[int]) -> str:
        self.decoded_ids = list(ids)
        return "decoded:" + ",".join(map(str, ids))

    def piece_to_id(self, piece: str) -> int:
        return self.pieces.get(piece, self.unk_id())

    def id_to_piece(self, token_id: int) -> str:
        return next((piece for piece, value in self.pieces.items() if value == token_id), "<unk>")


def test_tokenizer_wrapper_forwards_to_processor_without_a_real_binary(monkeypatch, tmp_path):
    tokenizer_path = tmp_path / "byteseed.model"
    tokenizer_path.touch()
    monkeypatch.setattr(tokenizer_module.spm, "SentencePieceProcessor", FakeSentencePieceProcessor)

    tokenizer = ByteSeedTokenizer(tmp_path)

    assert tokenizer.encode("hello", add_bos=True, add_eos=True) == [1, 6, 7, 2]
    assert tokenizer.decode([6, 99, -1]) == "decoded:6,0,0"
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.vocab_size == 10
    assert marker_id(tokenizer, "<|end|>") == 4
    assert marker_id(tokenizer, "missing") is None
    assert stop_token_ids(tokenizer, stop_at_end=True) == {4, 5}


def test_missing_tokenizer_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Tokenizer file missing"):
        ByteSeedTokenizer(tmp_path)
