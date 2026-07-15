from __future__ import annotations

import numpy as np
import pytest
import torch

import byteseed.dataset as dataset_module
from byteseed.dataset import TokenDataset


@pytest.mark.parametrize("token_count", [0, 1, 4])
def test_token_dataset_rejects_insufficient_tokens(token_count):
    with pytest.raises(ValueError, match="too short"):
        TokenDataset(np.arange(token_count, dtype=np.int64), block_size=4, device="cpu")


def test_token_dataset_accepts_its_minimum_documented_length():
    tokens = np.arange(5, dtype=np.int64)
    dataset = TokenDataset(tokens, block_size=4, device="cpu")

    x, y = dataset.get_batch(batch_size=1)

    assert x.shape == (1, 4)
    assert y.shape == (1, 4)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(x[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y[0], torch.tensor([1, 2, 3, 4]))


def test_token_dataset_includes_every_start_at_one_token_beyond_minimum(monkeypatch):
    dataset = TokenDataset(np.arange(6, dtype=np.int64), block_size=4, device="cpu")

    def fixed_starts(high: int, size: tuple[int, ...]) -> torch.Tensor:
        assert high == 2
        assert size == (2,)
        return torch.tensor([0, 1], dtype=torch.long)

    monkeypatch.setattr(dataset_module.torch, "randint", fixed_starts)
    x, y = dataset.get_batch(batch_size=2)

    assert torch.equal(x, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert torch.equal(y, torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]))


def test_token_dataset_returns_shifted_windows_in_bounds():
    dataset = TokenDataset(np.arange(20, dtype=np.int64), block_size=4, device="cpu")

    x, y = dataset.get_batch(batch_size=3)

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(y, x + 1)


def test_token_dataset_repeated_sampling_stays_within_data_bounds():
    dataset = TokenDataset(np.arange(20, dtype=np.int64), block_size=4, device="cpu")

    for _ in range(20):
        x, y = dataset.get_batch(batch_size=2)
        assert int(x.min()) >= 0
        assert int(y.max()) < 20
        assert torch.equal(y, x + 1)
