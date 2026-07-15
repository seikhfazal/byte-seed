from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    """Reset every supported RNG so tests do not depend on execution order."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)


@pytest.fixture
def tiny_config():
    from byteseed.config import ByteSeedConfig

    return ByteSeedConfig(
        model_name="ByteSeed-Test",
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
    )


@pytest.fixture
def tiny_model(tiny_config):
    from byteseed.model import GPT

    model = GPT(tiny_config)
    model.eval()
    return model
