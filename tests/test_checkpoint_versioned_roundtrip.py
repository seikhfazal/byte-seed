from __future__ import annotations

import torch

from byteseed.checkpoint import CheckpointKind, build_checkpoint
from byteseed.generate import load_model


def test_versioned_sft_checkpoint_round_trip_preserves_model_state(tmp_path, tiny_config, tiny_model):
    checkpoint_path = tmp_path / "versioned-sft.pt"
    original_state = tiny_model.state_dict()
    torch.save(
        build_checkpoint(
            CheckpointKind.SFT,
            model_state=original_state,
            config=tiny_config.__dict__,
            iteration=9,
        ),
        checkpoint_path,
    )

    loaded = load_model(tiny_config, str(checkpoint_path))
    loaded_state = loaded.state_dict()

    assert loaded.config == tiny_config
    assert loaded_state.keys() == original_state.keys()
    for name, tensor in original_state.items():
        assert torch.equal(loaded_state[name], tensor), name


def test_versioned_pretrain_checkpoint_is_valid_for_inference(tmp_path, tiny_config, tiny_model):
    checkpoint_path = tmp_path / "versioned-pretrain.pt"
    torch.save(
        build_checkpoint(
            CheckpointKind.PRETRAIN,
            model_state=tiny_model.state_dict(),
            optimizer_state={"state": {}, "param_groups": []},
            config=tiny_config.__dict__,
            iteration=3,
        ),
        checkpoint_path,
    )

    loaded = load_model(tiny_config, str(checkpoint_path))

    assert next(loaded.parameters()).device.type == "cpu"
    assert loaded.token_embedding.weight.shape == tiny_model.token_embedding.weight.shape
