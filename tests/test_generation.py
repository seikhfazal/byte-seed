from __future__ import annotations

import torch


def _always_generate(token_id: int):
    def forward(idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time = idx.shape
        logits = torch.full((batch, time, 32), -1e9, dtype=torch.float32, device=idx.device)
        logits[:, :, token_id] = 0.0
        return logits, None

    return forward


def _scheduled_generate(schedule: list[list[int]], prompt_length: int = 2):
    def forward(idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time = idx.shape
        step = min(time - prompt_length, len(schedule) - 1)
        tokens = schedule[step]
        assert len(tokens) == batch
        logits = torch.full((batch, time, 32), -1e9, dtype=torch.float32, device=idx.device)
        for row, token_id in enumerate(tokens):
            logits[row, :, token_id] = 0.0
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


def test_stop_tokens_finish_rows_independently_and_preserve_batch_alignment(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _scheduled_generate([[4, 5], [9, 4]]))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=3, stop_token_ids={4})

    assert generated.shape == (2, 4)
    assert generated.dtype == prompt.dtype
    assert generated.device == prompt.device
    assert torch.equal(generated, torch.tensor([[1, 2, 4, 4], [3, 8, 5, 4]]))


def test_finished_rows_use_inert_stop_fillers_while_active_rows_reach_limit(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _scheduled_generate([[4, 5], [9, 6], [8, 7]]))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=3, stop_token_ids={4})

    assert generated.shape == (2, 5)
    assert torch.equal(generated[0], torch.tensor([1, 2, 4, 4, 4]))
    assert torch.equal(generated[1], torch.tensor([3, 8, 5, 6, 7]))


def test_multiple_stop_token_ids_finish_only_the_emitting_row(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _scheduled_generate([[5, 6], [8, 4]]))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=3, stop_token_ids={4, 5, 5})

    assert generated.shape == (2, 4)
    assert torch.equal(generated, torch.tensor([[1, 2, 5, 5], [3, 8, 6, 4]]))


def test_generation_without_stop_tokens_uses_the_full_generation_budget(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _scheduled_generate([[4, 5], [6, 7]]))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=2)

    assert generated.shape == (2, 4)
    assert torch.equal(generated, torch.tensor([[1, 2, 4, 6], [3, 8, 5, 7]]))


def test_empty_stop_token_set_uses_the_full_generation_budget(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _scheduled_generate([[4, 5], [6, 7]]))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=2, stop_token_ids=set())

    assert generated.shape == (2, 4)
    assert torch.equal(generated, torch.tensor([[1, 2, 4, 6], [3, 8, 5, 7]]))


def test_all_rows_stopping_on_the_same_step_exit_immediately(tiny_model, monkeypatch):
    monkeypatch.setattr(tiny_model, "forward", _always_generate(4))
    prompt = torch.tensor([[1, 2], [3, 8]], dtype=torch.long)

    generated = tiny_model.generate(prompt, max_new_tokens=3, stop_token_ids={4})

    assert generated.shape == (2, 3)
    assert torch.equal(generated[:, -1], torch.tensor([4, 4]))