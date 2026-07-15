from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from byteseed.checkpoint import (
    RESUME_STATE_VERSION,
    CheckpointCompatibilityError,
    CheckpointKind,
    CheckpointValidationError,
    build_checkpoint,
    build_resume_state,
    capture_rng_state,
    capture_scaler_state,
    is_exact_resumable,
    move_optimizer_state_to_device,
    restore_rng_state,
    restore_scaler_state,
    training_config_snapshot,
    validate_exact_resume_checkpoint,
    validate_training_config,
)


class FakeScaler:
    def __init__(self, enabled: bool, state: Mapping[str, Any] | None = None):
        self._enabled = enabled
        self._state = dict(state or ({"scale": 128.0} if enabled else {}))
        self.loaded: dict[str, Any] | None = None

    def is_enabled(self) -> bool:
        return self._enabled

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.loaded = copy.deepcopy(dict(state))
        self._state = copy.deepcopy(dict(state))


def _critical_config(tiny_config) -> dict[str, Any]:
    return training_config_snapshot(
        tiny_config.__dict__,
        device_type="cpu",
        amp_enabled=False,
    )


def _exact_checkpoint(tiny_config, checkpoint_provenance) -> dict[str, Any]:
    critical = _critical_config(tiny_config)
    resume_state = build_resume_state(
        scaler=FakeScaler(False),
        best_val=0.25,
        patience_left=2,
        training_config=critical,
    )
    return build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"state": {}, "param_groups": []},
        config=tiny_config.__dict__,
        iteration=4,
        best_val=0.25,
        resume_state=resume_state,
        provenance=checkpoint_provenance,
    )


def test_new_pretrain_checkpoint_contains_versioned_exact_resume_state(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)

    state = validate_exact_resume_checkpoint(checkpoint)

    assert state["version"] == RESUME_STATE_VERSION
    assert set(state) == {
        "version",
        "rng_state",
        "amp_scaler",
        "early_stopping",
        "training_config",
    }
    assert set(state["rng_state"]) == {"python", "torch_cpu", "torch_cuda"}
    assert state["amp_scaler"] == {"enabled": False, "state": {}}
    assert state["early_stopping"] == {"best_val": 0.25, "patience_left": 2}
    assert is_exact_resumable(checkpoint)


@pytest.mark.parametrize("kind", [CheckpointKind.SFT, CheckpointKind.MODEL_ONLY])
def test_non_pretrain_checkpoint_is_not_exact_resumable(kind, tiny_config):
    checkpoint = build_checkpoint(
        kind,
        model_state={"weight": torch.tensor([1.0])},
        config=tiny_config.__dict__,
        iteration=1 if kind is CheckpointKind.SFT else None,
    )

    assert not is_exact_resumable(checkpoint)


def test_future_resume_state_version_fails_clearly(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    checkpoint["resume_state"]["version"] = RESUME_STATE_VERSION + 1

    with pytest.raises(CheckpointValidationError, match="Unsupported resume-state version"):
        validate_exact_resume_checkpoint(checkpoint)


def test_missing_required_resume_field_fails_clearly(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    del checkpoint["resume_state"]["rng_state"]

    with pytest.raises(CheckpointValidationError, match="missing required fields: rng_state"):
        validate_exact_resume_checkpoint(checkpoint)


def test_missing_critical_training_configuration_fails_exact_validation(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    del checkpoint["resume_state"]["training_config"]["batch_size"]

    with pytest.raises(CheckpointValidationError, match="training_config is incomplete.*batch_size"):
        validate_exact_resume_checkpoint(checkpoint)

def test_exact_resume_requires_structural_optimizer_state(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    del checkpoint["optimizer"]

    assert not is_exact_resumable(checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="complete structural state.*optimizer"):
        validate_exact_resume_checkpoint(checkpoint)

def test_partial_pretrain_checkpoint_is_not_exact_resumable(tiny_config):
    checkpoint = build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"state": {}, "param_groups": []},
        config=tiny_config.__dict__,
        iteration=4,
        best_val=0.25,
    )

    assert not is_exact_resumable(checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="lacks exact resume_state"):
        validate_exact_resume_checkpoint(checkpoint)


def test_python_rng_sequence_matches_after_restore():
    random.seed(901)
    state = capture_rng_state()
    expected = [random.random() for _ in range(5)]

    for _ in range(7):
        random.random()
    restore_rng_state(state)

    assert [random.random() for _ in range(5)] == expected


def test_torch_cpu_rng_sequence_matches_after_restore():
    torch.manual_seed(902)
    state = capture_rng_state()
    expected = torch.rand(6)

    torch.rand(11)
    restore_rng_state(state)

    assert torch.equal(torch.rand(6), expected)


def test_cpu_only_capture_does_not_query_or_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_call():
        raise AssertionError("CUDA initialization/state query should not occur")

    monkeypatch.setattr(torch.cuda, "is_initialized", unexpected_call)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_call)

    state = capture_rng_state()

    assert state["torch_cuda"] is None


def test_cuda_all_device_state_capture_and_restore_with_doubles(monkeypatch):
    saved_cuda = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3], dtype=torch.uint8)]
    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: saved_cuda)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: restored.append([value.clone() for value in values]),
    )

    state = capture_rng_state()
    restore_rng_state(state)

    assert len(restored) == 1
    assert all(torch.equal(actual, expected) for actual, expected in zip(restored[0], saved_cuda))


def test_cuda_device_count_mismatch_fails_before_restore(monkeypatch):
    state = capture_rng_state()
    state["torch_cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _values: pytest.fail("mismatched CUDA state must not be applied"),
    )

    with pytest.raises(CheckpointCompatibilityError, match="device-count mismatch"):
        restore_rng_state(state)

def test_cuda_absent_resume_state_restores_without_cuda(monkeypatch):
    state = capture_rng_state()
    state["torch_cuda"] = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _values: pytest.fail("CUDA restore should not be called"),
    )

    restore_rng_state(state)


def test_active_scaler_state_round_trip():
    source = FakeScaler(True, {"scale": 512.0, "growth_tracker": 3})
    saved = capture_scaler_state(source)
    target = FakeScaler(True, {"scale": 1.0})

    restore_scaler_state(target, saved)

    assert target.loaded == {"scale": 512.0, "growth_tracker": 3}


def test_disabled_scaler_state_is_represented_cleanly():
    saved = capture_scaler_state(FakeScaler(False))

    restore_scaler_state(FakeScaler(False), saved)

    assert saved == {"enabled": False, "state": {}}


def test_scaler_enablement_mismatch_fails_clearly():
    with pytest.raises(CheckpointCompatibilityError, match="AMP configuration mismatch"):
        restore_scaler_state(FakeScaler(False), {"enabled": True, "state": {"scale": 2.0}})


def test_missing_active_scaler_state_cannot_validate_as_exact(tiny_config, checkpoint_provenance):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    checkpoint["resume_state"]["amp_scaler"] = {"enabled": True, "state": {}}

    with pytest.raises(CheckpointValidationError, match="requires a non-empty saved state"):
        validate_exact_resume_checkpoint(checkpoint)


def test_matching_training_configuration_is_accepted(tiny_config):
    saved = _critical_config(tiny_config)
    current = dict(saved)
    current["presentation_only"] = "ignored"

    validate_training_config(saved, current)


def test_training_configuration_mismatch_names_changed_field(tiny_config):
    saved = _critical_config(tiny_config)
    current = dict(saved)
    current["batch_size"] += 1

    with pytest.raises(CheckpointCompatibilityError, match=r"batch_size .*checkpoint=.*current="):
        validate_training_config(saved, current)


def test_optimizer_state_device_migration_handles_nested_values():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "nested": {
            "list": [torch.tensor([3.0]), "unchanged"],
            "tuple": (torch.tensor([4.0]), 5),
        },
    }

    move_optimizer_state_to_device(optimizer, "cpu")

    state = optimizer.state[parameter]
    assert state["step"].device.type == "cpu"
    assert state["nested"]["list"][0].device.type == "cpu"
    assert state["nested"]["list"][1] == "unchanged"
    assert state["nested"]["tuple"][0].device.type == "cpu"
    assert state["nested"]["tuple"][1] == 5


class TinyStochasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.35)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(inputs))


def _make_training_objects() -> tuple[TinyStochasticModel, torch.optim.AdamW]:
    model = TinyStochasticModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    return model, optimizer


def _tiny_step(
    model: TinyStochasticModel,
    optimizer: torch.optim.AdamW,
) -> tuple[float, torch.Tensor]:
    python_sample = random.random()
    inputs = torch.randn(3, 4)
    target = torch.full((3, 4), python_sample)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    return python_sample, inputs.clone()


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_interrupted_resume_matches_uninterrupted_cpu_continuation(
    tmp_path, tiny_config, checkpoint_provenance
):
    first_steps = 3
    remaining_steps = 2
    critical = _critical_config(tiny_config)

    random.seed(77)
    torch.manual_seed(77)
    control_model, control_optimizer = _make_training_objects()
    control_samples = [
        _tiny_step(control_model, control_optimizer)
        for _ in range(first_steps + remaining_steps)
    ]
    control_python_next = random.random()
    control_torch_next = torch.rand(4)

    random.seed(77)
    torch.manual_seed(77)
    interrupted_model, interrupted_optimizer = _make_training_objects()
    interrupted_samples = [
        _tiny_step(interrupted_model, interrupted_optimizer)
        for _ in range(first_steps)
    ]
    resume_state = build_resume_state(
        scaler=FakeScaler(False),
        best_val=0.4,
        patience_left=2,
        training_config=critical,
    )
    checkpoint = build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=interrupted_model.state_dict(),
        optimizer_state=interrupted_optimizer.state_dict(),
        config=tiny_config.__dict__,
        iteration=first_steps - 1,
        best_val=0.4,
        resume_state=resume_state,
        provenance=checkpoint_provenance,
    )
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(checkpoint, checkpoint_path)

    for _ in range(9):
        random.random()
    torch.rand(13)

    resumed_model, resumed_optimizer = _make_training_objects()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resumed_model.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    move_optimizer_state_to_device(resumed_optimizer, "cpu")
    loaded_resume = validate_exact_resume_checkpoint(loaded)
    restore_scaler_state(FakeScaler(False), loaded_resume["amp_scaler"])
    validate_training_config(loaded_resume["training_config"], critical)
    restore_rng_state(loaded_resume["rng_state"])

    resumed_samples = [
        _tiny_step(resumed_model, resumed_optimizer)
        for _ in range(remaining_steps)
    ]
    resumed_python_next = random.random()
    resumed_torch_next = torch.rand(4)

    for name, parameter in control_model.state_dict().items():
        assert torch.equal(parameter, resumed_model.state_dict()[name]), name
    _assert_nested_equal(control_optimizer.state_dict(), resumed_optimizer.state_dict())
    for interrupted, control in zip(interrupted_samples, control_samples[:first_steps]):
        assert interrupted[0] == control[0]
        assert torch.equal(interrupted[1], control[1])
    for resumed, control in zip(resumed_samples, control_samples[first_steps:]):
        assert resumed[0] == control[0]
        assert torch.equal(resumed[1], control[1])
    assert resumed_python_next == control_python_next
    assert torch.equal(resumed_torch_next, control_torch_next)
    assert loaded["iter"] + 1 == first_steps
    assert loaded_resume["early_stopping"] == {"best_val": 0.4, "patience_left": 2}
