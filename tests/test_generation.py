from __future__ import annotations

import pytest
import torch


def _always_generate(token_id: int):
    def forward(idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time = idx.shape
        logits = torch.full((batch, time, 32), -1e9, dtype=torch.float32, device=idx.device)
        logits[:, :, token_id] = 0.0
        return logits, None

    return forward


def test_generation_is_repeatable_with_fixed_seed_and_preserves_prefix(tiny_model):
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    torch.manual_seed(77)
    first = tiny_model.generate(prompt, max_new_tokens=4, top_k=1)
    torch.manual_seed(77)
    second = tiny_model.generate(prompt, max_new_tokens=4, top_k=1)

    assert torch.equal(first, second)
    assert torch.equal(first[:, : prompt.size(1)], prompt)
    assert first.shape == (1, prompt.size(1) + 4)


def test_generation_respects_vocab_limit_and_repetition_penalty(tiny_model):
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    generated = tiny_model.generate(
        prompt,
        max_new_tokens=3,
        top_k=1,
        vocab_limit=5,
        repetition_penalty=1.2,
    )

    assert generated.shape == (1, 5)
    assert int(generated[:, prompt.size(1) :].max()) < 5


def test_generation_handles_zero_temperature_and_large_top_k(tiny_model):
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=1, temperature=0.0, top_k=100)

    assert generated.shape == (1, 3)


def test_stop_token_stops_batch_size_one(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _always_generate(4))
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=3, stop_token_ids={4})

    assert generated.shape == (1, 3)
    assert generated[0, -1].item() == 4


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="Known v0.4 audit defect: batched generation ignores stop tokens")
def test_stop_tokens_stop_every_sequence_in_a_batch(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _always_generate(4))
    prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=2, stop_token_ids={4})

    assert generated.shape == (2, 3)
