from __future__ import annotations

import numpy as np
import pytest
import torch

from byteseed.dataset import TokenDataset


def test_token_dataset_returns_shifted_windows_in_bounds():
    dataset = TokenDataset(np.arange(20, dtype=np.int64), block_size=4, device="cpu")

    x, y = dataset.get_batch(batch_size=3)

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert torch.equal(y, x + 1)


def test_token_dataset_repeated_sampling_stays_within_data_bounds():
    dataset = TokenDataset(np.arange(20, dtype=np.int64), block_size=4, device="cpu")

    for _ in range(20):
        x, y = dataset.get_batch(batch_size=2)
        assert int(x.min()) >= 0
        assert int(y.max()) < 20


def test_token_dataset_rejects_insufficient_tokens():
    with pytest.raises(ValueError, match="too short"):
        TokenDataset(np.arange(4, dtype=np.int64), block_size=4, device="cpu")


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="Known v0.4 audit defect: TokenDataset fails at block_size plus one tokens")
def test_token_dataset_accepts_its_minimum_documented_length():
    dataset = TokenDataset(np.arange(5, dtype=np.int64), block_size=4, device="cpu")

    x, y = dataset.get_batch(batch_size=1)

    assert x.shape == (1, 4)
    assert y.shape == (1, 4)
