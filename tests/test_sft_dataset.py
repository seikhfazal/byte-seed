from __future__ import annotations

import json

import pytest

from byteseed.finetune_chat import ChatSFTDataset, IGNORE_INDEX, format_chat


class CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        base = 10 if text.startswith("<|user|>") else 1000
        ids = list(range(base, base + len(text)))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids


def _write_example(path, user: str, assistant: str) -> None:
    path.write_text(json.dumps({"user": user, "assistant": assistant}) + "\n", encoding="utf-8")


def _load_example(tmp_path, user: str, assistant: str, block_size: int) -> tuple[ChatSFTDataset, CharacterTokenizer]:
    example_path = tmp_path / "examples.jsonl"
    tokenizer = CharacterTokenizer()
    _write_example(example_path, user, assistant)
    return ChatSFTDataset(example_path, tokenizer, block_size=block_size, device="cpu"), tokenizer


def _supervised_targets(labels: list[int]) -> list[int]:
    return [label for label in labels if label != IGNORE_INDEX]


def test_short_sft_example_preserves_format_and_exact_masking(tmp_path):
    dataset, tokenizer = _load_example(tmp_path, "Hi", "Hello", block_size=64)
    x, y = dataset.examples[0]
    prompt_ids = tokenizer.encode("<|user|>\nHi\n<|assistant|>\n", add_bos=True)
    answer_ids = tokenizer.encode("Hello\n<|end|>", add_bos=False)

    assert format_chat("Hi", "Hello") == "<|user|>\nHi\n<|assistant|>\nHello\n<|end|>"
    assert x == (prompt_ids + answer_ids)[:-1]
    assert y[: len(prompt_ids) - 1] == [IGNORE_INDEX] * (len(prompt_ids) - 1)
    assert y[len(prompt_ids) - 1 :] == answer_ids
    assert len(x) == len(y)


@pytest.mark.parametrize(
    ("user", "assistant"),
    [
        ("u" * 100, "OK"),
        ("Hi", "a" * 100),
        ("u" * 100, "a" * 100),
    ],
)
def test_sft_truncation_cases_preserve_assistant_supervision(tmp_path, user, assistant):
    block_size = 32
    dataset, tokenizer = _load_example(tmp_path, user, assistant, block_size)
    x, y = dataset.examples[0]
    answer_ids = tokenizer.encode(f"{assistant}\n<|end|>", add_bos=False)
    supervised = _supervised_targets(y)

    assert len(x) == block_size
    assert len(y) == block_size
    assert y[0] == IGNORE_INDEX
    assert supervised
    assert all(label in answer_ids for label in supervised)
    assert supervised[-1] == answer_ids[-1]


def test_short_prompt_is_preserved_when_only_answer_is_overlong(tmp_path):
    block_size = 32
    assistant = "a" * 100
    dataset, tokenizer = _load_example(tmp_path, "Hi", assistant, block_size)
    x, y = dataset.examples[0]
    prompt_ids = tokenizer.encode("<|user|>\nHi\n<|assistant|>\n", add_bos=True)
    answer_ids = tokenizer.encode(f"{assistant}\n<|end|>", add_bos=False)
    supervised = _supervised_targets(y)

    assert x[: len(prompt_ids)] == prompt_ids
    assert y[: len(prompt_ids) - 1] == [IGNORE_INDEX] * (len(prompt_ids) - 1)
    assert supervised
    assert all(label in answer_ids for label in supervised)
    assert supervised[-1] == answer_ids[-1]


def test_long_prompt_retains_at_least_one_supervised_target(tmp_path):
    block_size = 8
    dataset, tokenizer = _load_example(tmp_path, "u" * 100, "answer", block_size)
    x, y = dataset.examples[0]
    answer_ids = tokenizer.encode("answer\n<|end|>", add_bos=False)
    supervised = _supervised_targets(y)
    _, batch_y = dataset.get_batch(batch_size=1)

    assert len(x) == block_size
    assert len(y) == block_size
    assert y[0] == IGNORE_INDEX
    assert supervised
    assert all(label in answer_ids for label in supervised)
    assert supervised[-1] == answer_ids[-1]
    assert (batch_y != IGNORE_INDEX).any()


def test_answer_that_nearly_fills_block_preserves_content_and_end_token(tmp_path):
    block_size = 16
    assistant = "a" * 7
    dataset, tokenizer = _load_example(tmp_path, "Hi", assistant, block_size)
    _, y = dataset.examples[0]
    answer_ids = tokenizer.encode(f"{assistant}\n<|end|>", add_bos=False)
    supervised = _supervised_targets(y)

    assert len(answer_ids) == block_size - 1
    assert y[0] == IGNORE_INDEX
    assert supervised == answer_ids


def test_sft_batch_masks_padding_and_retains_supervision(tmp_path):
    dataset, _ = _load_example(tmp_path, "Hi", "OK", block_size=64)

    x, y = dataset.get_batch(batch_size=1)

    example_length = len(dataset.examples[0][1])
    assert x.shape == (1, 64)
    assert y.shape == (1, 64)
    assert (y != IGNORE_INDEX).any()
    assert (y[0, example_length:] == IGNORE_INDEX).all()


def test_impossible_block_size_reports_non_private_token_context(tmp_path):
    example_path = tmp_path / "examples.jsonl"
    _write_example(example_path, "private user text", "private assistant text")

    with pytest.raises(
        ValueError,
        match=r"line 1.*no supervised assistant target tokens.*block_size=0.*prompt_tokens=\d+.*answer_tokens=\d+",
    ) as error:
        ChatSFTDataset(example_path, CharacterTokenizer(), block_size=0, device="cpu")

    assert "private user text" not in str(error.value)
    assert "private assistant text" not in str(error.value)
