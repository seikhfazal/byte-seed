from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class ByteSeedTokenizer:
    def __init__(self, tokenizer_dir: str | Path):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.model_path = self.tokenizer_dir / "byteseed.model"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file missing: {self.model_path}\n"
                "Run: python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml"
            )
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        unk = self.sp.unk_id()
        safe_ids = [token_id if 0 <= token_id < self.vocab_size else unk for token_id in ids]
        return self.sp.decode(safe_ids)


