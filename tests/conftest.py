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


@pytest.fixture
def tokenizer_identity(tmp_path):
    from byteseed.provenance import REQUIRED_SPECIAL_TOKENS, create_tokenizer_identity

    model_path = tmp_path / "byteseed.model"
    model_path.write_bytes(b"synthetic sentencepiece model bytes")
    special_tokens = {token: index for index, token in enumerate(REQUIRED_SPECIAL_TOKENS)}
    return create_tokenizer_identity(
        model_path,
        vocab_size=32,
        special_tokens=special_tokens,
    )


@pytest.fixture
def data_manifest(tmp_path, tokenizer_identity):
    from byteseed.provenance import build_pretraining_data_manifest

    np.save(tmp_path / "train.npy", np.arange(24, dtype=np.uint16))
    np.save(tmp_path / "val.npy", np.arange(8, dtype=np.uint16))
    return build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.75,
    )


@pytest.fixture
def checkpoint_provenance(tokenizer_identity, data_manifest):
    from byteseed.provenance import build_checkpoint_provenance

    return build_checkpoint_provenance(
        tokenizer_identity,
        data_manifest=data_manifest,
    )
