from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from byteseed.generate import load_model


def _assert_same_state(first, second) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_current_checkpoint_round_trip_on_cpu(tmp_path, tiny_config, tiny_model):
    checkpoint_path = tmp_path / "current.pt"
    torch.save({"model": tiny_model.state_dict(), "config": tiny_config.__dict__, "iter": 3}, checkpoint_path)

    raw = torch.load(checkpoint_path, map_location="cpu")
    loaded = load_model(tiny_config, str(checkpoint_path))

    assert raw["model"]["token_embedding.weight"].device.type == "cpu"
    assert loaded.config.vocab_size == tiny_config.vocab_size
    assert loaded.config.block_size == tiny_config.block_size
    assert loaded.token_embedding.weight.shape == tiny_model.token_embedding.weight.shape
    _assert_same_state(tiny_model, loaded)


def test_legacy_checkpoint_without_config_remains_loadable(tmp_path, tiny_config, tiny_model):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model": tiny_model.state_dict()}, checkpoint_path)

    loaded = load_model(tiny_config, str(checkpoint_path))

    assert next(loaded.parameters()).device.type == "cpu"
    _assert_same_state(tiny_model, loaded)

def test_legacy_inference_with_runtime_tokenizer_warns_as_unverified(
    tmp_path, tiny_config, tiny_model, tokenizer_identity
):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model": tiny_model.state_dict()}, checkpoint_path)
    tokenizer = SimpleNamespace(identity=tokenizer_identity)

    with pytest.warns(RuntimeWarning, match="legacy inference compatibility is unverified"):
        loaded = load_model(tiny_config, str(checkpoint_path), tokenizer=tokenizer)

    _assert_same_state(tiny_model, loaded)
