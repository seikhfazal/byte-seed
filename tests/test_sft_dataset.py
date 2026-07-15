from __future__ import annotations

import json

import pytest

from byteseed.finetune_chat import ChatSFTDataset, IGNORE_INDEX, format_chat


class CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [10 + (ord(character) % 20) for character in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids


def _write_example(path, user: str, assistant: str) -> None:
    path.write_text(json.dumps({"user": user, "assistant": assistant}) + "\n", encoding="utf-8")


def test_sft_format_and_masking_keep_assistant_labels(tmp_path):
    example_path = tmp_path / "examples.jsonl"
    tokenizer = CharacterTokenizer()
    _write_example(example_path, "Hi", "Hello")

    dataset = ChatSFTDataset(example_path, tokenizer, block_size=64, device="cpu")
    x, y = dataset.examples[0]
    prompt_ids = tokenizer.encode("<|user|>\nHi\n<|assistant|>\n", add_bos=True)

    assert format_chat("Hi", "Hello") == "<|user|>\nHi\n<|assistant|>\nHello\n<|end|>"
    assert y[: len(prompt_ids) - 1] == [IGNORE_INDEX] * (len(prompt_ids) - 1)
    assert any(label != IGNORE_INDEX for label in y)
    assert len(x) == len(y)


def test_sft_batch_masks_padding(tmp_path):
    example_path = tmp_path / "examples.jsonl"
    _write_example(example_path, "Hi", "OK")
    dataset = ChatSFTDataset(example_path, CharacterTokenizer(), block_size=64, device="cpu")

    x, y = dataset.get_batch(batch_size=1)

    supervised = int((y != IGNORE_INDEX).sum().item())
    assert x.shape == (1, 64)
    assert y.shape == (1, 64)
    assert supervised > 0
    assert (y[0, len(dataset.examples[0][1]) :] == IGNORE_INDEX).all()


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="Known v0.4 audit defect: SFT truncation can remove all assistant targets")
def test_long_prompt_retains_at_least_one_supervised_target(tmp_path):
    example_path = tmp_path / "examples.jsonl"
    _write_example(example_path, "u" * 100, "answer")

    dataset = ChatSFTDataset(example_path, CharacterTokenizer(), block_size=8, device="cpu")

    assert any(label != IGNORE_INDEX for label in dataset.examples[0][1])
