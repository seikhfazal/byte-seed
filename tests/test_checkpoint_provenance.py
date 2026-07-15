from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from byteseed.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointKind,
    CheckpointOperation,
    CheckpointValidationError,
    build_checkpoint,
    build_resume_state,
    discover_checkpoint,
    load_checkpoint,
    select_checkpoint,
    training_config_snapshot,
    validate_exact_resume_checkpoint,
    validate_state_complete_checkpoint,
)
from byteseed.pretrain import resolve_resume_checkpoint
from byteseed.provenance import (
    REQUIRED_SPECIAL_TOKENS,
    build_checkpoint_provenance,
    build_pretraining_data_manifest,
    create_data_manifest,
    create_tokenizer_identity,
)


def _model_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([1.0])}


def _optimizer_state() -> dict[str, object]:
    return {"state": {}, "param_groups": []}


def _exact_checkpoint(tiny_config, provenance, iteration: int = 4):
    critical = training_config_snapshot(
        tiny_config.__dict__,
        device_type="cpu",
        amp_enabled=False,
    )
    resume_state = build_resume_state(
        scaler=None,
        best_val=0.5,
        patience_left=2,
        training_config=critical,
    )
    return build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config=tiny_config.__dict__,
        iteration=iteration,
        best_val=0.5,
        resume_state=resume_state,
        provenance=provenance,
    )


def _alternate_tokenizer_identity(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "byteseed.model"
    model.write_bytes(b"same-sized alternate tokenizer")
    return create_tokenizer_identity(
        model,
        vocab_size=32,
        special_tokens={
            token: index for index, token in enumerate(REQUIRED_SPECIAL_TOKENS)
        },
    )


def _manifest_with_changed_artifact(data_manifest):
    artifacts = copy.deepcopy(data_manifest["artifacts"])
    artifacts[0]["sha256"] = "f" * 64
    return create_data_manifest(
        tokenizer_identity=data_manifest["tokenizer"],
        artifacts=artifacts,
        preprocessing=data_manifest["preprocessing"],
    )


def _manifest_with_changed_split(data_manifest):
    preprocessing = copy.deepcopy(data_manifest["preprocessing"])
    preprocessing["split"]["train_fraction"] = 0.8
    return create_data_manifest(
        tokenizer_identity=data_manifest["tokenizer"],
        artifacts=data_manifest["artifacts"],
        preprocessing=preprocessing,
    )


def _save(path: Path, checkpoint) -> Path:
    torch.save(checkpoint, path)
    return path


def test_checkpoint_kinds_store_only_honest_provenance(
    tiny_config, tokenizer_identity, data_manifest, checkpoint_provenance
):
    pretrain = build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config=tiny_config.__dict__,
        iteration=1,
        provenance=checkpoint_provenance,
    )
    tokenizer_only = build_checkpoint_provenance(tokenizer_identity)
    sft = build_checkpoint(
        CheckpointKind.SFT,
        model_state=_model_state(),
        config=tiny_config.__dict__,
        iteration=1,
        provenance=tokenizer_only,
    )
    model_only = build_checkpoint(
        CheckpointKind.MODEL_ONLY,
        model_state=_model_state(),
        config=tiny_config.__dict__,
        provenance=tokenizer_only,
    )

    assert pretrain["checkpoint_version"] == 1
    assert pretrain["provenance"]["data_manifest"]["digest"] == data_manifest["digest"]
    assert pretrain["provenance"]["data_manifest_digest"] == data_manifest["digest"]
    assert sft["provenance"] == tokenizer_only
    assert model_only["provenance"] == tokenizer_only
    assert "data_manifest" not in sft["provenance"]

    with pytest.raises(ValueError, match="tokenizer provenance only"):
        build_checkpoint(
            CheckpointKind.SFT,
            model_state=_model_state(),
            config=tiny_config.__dict__,
            iteration=1,
            provenance=checkpoint_provenance,
        )


def test_matching_provenance_accepts_exact_resume(
    tmp_path, tiny_config, checkpoint_provenance
):
    path = _save(tmp_path / "exact.pt", _exact_checkpoint(tiny_config, checkpoint_provenance))

    loaded = load_checkpoint(
        path,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )

    assert loaded.path == path
    assert loaded.tokenizer_verified is True


def test_tokenizer_corpus_and_split_mismatches_reject_exact_resume(
    tmp_path, tiny_config, checkpoint_provenance
):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    alternate_tokenizer = _alternate_tokenizer_identity(tmp_path / "alternate")
    changed_corpus = _manifest_with_changed_artifact(checkpoint_provenance["data_manifest"])
    changed_split = _manifest_with_changed_split(checkpoint_provenance["data_manifest"])

    with pytest.raises(CheckpointCompatibilityError, match="tokenizer.model bytes"):
        validate_exact_resume_checkpoint(
            checkpoint,
            runtime_tokenizer_identity=alternate_tokenizer,
            runtime_data_manifest=changed_corpus,
        )
    with pytest.raises(CheckpointCompatibilityError, match="training corpus"):
        validate_exact_resume_checkpoint(
            checkpoint,
            runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
            runtime_data_manifest=changed_corpus,
        )
    with pytest.raises(CheckpointCompatibilityError, match="split configuration"):
        validate_exact_resume_checkpoint(
            checkpoint,
            runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
            runtime_data_manifest=changed_split,
        )


def test_pr4_state_complete_checkpoint_is_provenance_unverified(
    tiny_config, checkpoint_provenance
):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    del checkpoint["provenance"]

    validate_state_complete_checkpoint(checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="provenance-unverified"):
        validate_exact_resume_checkpoint(checkpoint)


def test_future_and_malformed_checkpoint_provenance_fail_explicitly(
    tmp_path, tiny_config, checkpoint_provenance
):
    future = _exact_checkpoint(tiny_config, checkpoint_provenance)
    future["provenance"]["version"] = 2
    future_path = _save(tmp_path / "future.pt", future)
    with pytest.raises(CheckpointValidationError, match="checkpoint provenance version"):
        load_checkpoint(future_path, CheckpointOperation.PRETRAIN_RESUME)

    malformed = _exact_checkpoint(tiny_config, checkpoint_provenance)
    malformed["provenance"] = "not-a-mapping"
    malformed_path = _save(tmp_path / "malformed.pt", malformed)
    with pytest.raises(CheckpointValidationError, match="provenance must be a mapping"):
        load_checkpoint(malformed_path, CheckpointOperation.PRETRAIN_RESUME)


@pytest.mark.parametrize("mismatch", ["tokenizer", "corpus"])
def test_auto_exact_resume_skips_newer_mismatched_checkpoint(
    tmp_path, tiny_config, checkpoint_provenance, mismatch
):
    matching_path = _save(
        tmp_path / "matching.pt",
        _exact_checkpoint(tiny_config, checkpoint_provenance, iteration=4),
    )
    if mismatch == "tokenizer":
        alternate_tokenizer = _alternate_tokenizer_identity(tmp_path / "alternate")

        mismatched = copy.deepcopy(checkpoint_provenance)
        mismatched["tokenizer"] = alternate_tokenizer
        mismatch_manifest = copy.deepcopy(mismatched["data_manifest"])
        mismatch_manifest["tokenizer"] = alternate_tokenizer
        mismatch_manifest["digest"] = create_data_manifest(
            tokenizer_identity=alternate_tokenizer,
            artifacts=mismatch_manifest["artifacts"],
            preprocessing=mismatch_manifest["preprocessing"],
        )["digest"]
        mismatched = build_checkpoint_provenance(
            alternate_tokenizer, data_manifest=mismatch_manifest
        )
    else:
        changed = _manifest_with_changed_artifact(checkpoint_provenance["data_manifest"])
        mismatched = build_checkpoint_provenance(
            checkpoint_provenance["tokenizer"], data_manifest=changed
        )
    _save(
        tmp_path / "mismatched.pt",
        _exact_checkpoint(tiny_config, mismatched, iteration=99),
    )

    selected = discover_checkpoint(
        tmp_path,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )

    assert selected is not None
    assert selected.path == matching_path


def test_corrupt_candidate_does_not_hide_matching_exact_checkpoint(
    tmp_path, tiny_config, checkpoint_provenance
):
    (tmp_path / "corrupt.pt").write_bytes(b"not a torch checkpoint")
    valid = _save(
        tmp_path / "valid.pt", _exact_checkpoint(tiny_config, checkpoint_provenance)
    )

    selected = discover_checkpoint(
        tmp_path,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )

    assert selected is not None
    assert selected.path == valid


def test_no_matching_exact_checkpoint_does_not_downgrade(
    tmp_path, tiny_config, checkpoint_provenance
):
    changed = _manifest_with_changed_artifact(checkpoint_provenance["data_manifest"])
    mismatched = build_checkpoint_provenance(
        checkpoint_provenance["tokenizer"], data_manifest=changed
    )
    _save(tmp_path / "mismatched.pt", _exact_checkpoint(tiny_config, mismatched))

    assert discover_checkpoint(
        tmp_path,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    ) is None


def test_explicit_mismatch_never_falls_back(
    tmp_path, tiny_config, checkpoint_provenance
):
    matching = _save(
        tmp_path / "matching.pt", _exact_checkpoint(tiny_config, checkpoint_provenance)
    )
    changed = _manifest_with_changed_artifact(checkpoint_provenance["data_manifest"])
    mismatched_provenance = build_checkpoint_provenance(
        checkpoint_provenance["tokenizer"], data_manifest=changed
    )
    mismatched = _save(
        tmp_path / "mismatched.pt",
        _exact_checkpoint(tiny_config, mismatched_provenance, iteration=99),
    )

    with pytest.raises(CheckpointCompatibilityError, match="training corpus"):
        select_checkpoint(
            tmp_path,
            CheckpointOperation.PRETRAIN_EXACT_RESUME,
            explicit_path=mismatched,
            runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
            runtime_data_manifest=checkpoint_provenance["data_manifest"],
        )
    assert matching.is_file()


def test_known_tokenizer_mismatch_is_forbidden_even_with_inexact_opt_in(
    tmp_path, tiny_config, checkpoint_provenance
):
    alternate_tokenizer = _alternate_tokenizer_identity(tmp_path / "alternate")
    alternate_manifest = create_data_manifest(
        tokenizer_identity=alternate_tokenizer,
        artifacts=checkpoint_provenance["data_manifest"]["artifacts"],
        preprocessing=checkpoint_provenance["data_manifest"]["preprocessing"],
    )
    alternate_provenance = build_checkpoint_provenance(
        alternate_tokenizer, data_manifest=alternate_manifest
    )
    path = _save(tmp_path / "alternate.pt", _exact_checkpoint(tiny_config, alternate_provenance))

    with pytest.raises(CheckpointCompatibilityError, match="Tokenizer identity mismatch"):
        resolve_resume_checkpoint(
            tmp_path,
            explicit_path=path,
            allow_inexact_resume=True,
            runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
            runtime_data_manifest=checkpoint_provenance["data_manifest"],
        )


def test_explicit_data_mismatch_can_only_continue_as_warned_inexact(
    tmp_path, tiny_config, checkpoint_provenance, capsys
):
    changed = _manifest_with_changed_artifact(checkpoint_provenance["data_manifest"])
    mismatched = build_checkpoint_provenance(
        checkpoint_provenance["tokenizer"], data_manifest=changed
    )
    path = _save(tmp_path / "changed-data.pt", _exact_checkpoint(tiny_config, mismatched))

    selected, resume_state = resolve_resume_checkpoint(
        tmp_path,
        explicit_path=path,
        allow_inexact_resume=True,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )

    assert selected is not None
    assert resume_state is None
    assert "not exact because" in capsys.readouterr().out


def test_legacy_inference_remains_compatible_but_unverified(
    tmp_path, tokenizer_identity
):
    path = _save(tmp_path / "anchor-like.pt", {"model": _model_state()})

    loaded = load_checkpoint(
        path,
        CheckpointOperation.MODEL_LOAD,
        runtime_tokenizer_identity=tokenizer_identity,
    )

    assert loaded.info.legacy
    assert loaded.tokenizer_verified is False


def test_inference_discovery_prefers_verified_match_over_newer_legacy(
    tmp_path, tiny_config, tokenizer_identity
):
    verified_path = _save(
        tmp_path / "verified.pt",
        build_checkpoint(
            CheckpointKind.MODEL_ONLY,
            model_state=_model_state(),
            config=tiny_config.__dict__,
            provenance=build_checkpoint_provenance(tokenizer_identity),
        ),
    )
    legacy_path = _save(tmp_path / "newer-legacy.pt", {"model": _model_state()})
    os.utime(verified_path, (1, 1))
    os.utime(legacy_path, (2, 2))

    selected = discover_checkpoint(
        tmp_path,
        CheckpointOperation.MODEL_LOAD,
        runtime_tokenizer_identity=tokenizer_identity,
    )

    assert selected is not None
    assert selected.path == verified_path

def test_reusing_precomputed_provenance_does_not_rehash_on_checkpoint_save(
    monkeypatch, tiny_config, checkpoint_provenance
):
    import byteseed.provenance as provenance_module

    monkeypatch.setattr(
        provenance_module,
        "sha256_file",
        lambda *_args, **_kwargs: pytest.fail("checkpoint save must reuse provenance"),
    )

    for iteration in (1, 2):
        checkpoint = build_checkpoint(
            CheckpointKind.PRETRAIN,
            model_state=_model_state(),
            optimizer_state=_optimizer_state(),
            config=tiny_config.__dict__,
            iteration=iteration,
            provenance=checkpoint_provenance,
        )
        assert checkpoint["provenance"]["data_manifest_digest"] == checkpoint_provenance[
            "data_manifest_digest"
        ]

def test_inference_rejects_known_tokenizer_mismatch_and_can_use_legacy_fallback(
    tmp_path, tiny_config, tokenizer_identity
):
    alternate_tokenizer = _alternate_tokenizer_identity(tmp_path / "alternate")
    mismatched_path = _save(
        tmp_path / "fingerprinted-mismatch.pt",
        build_checkpoint(
            CheckpointKind.MODEL_ONLY,
            model_state=_model_state(),
            config=tiny_config.__dict__,
            provenance=build_checkpoint_provenance(alternate_tokenizer),
        ),
    )
    legacy_path = _save(tmp_path / "legacy-anchor.pt", {"model": _model_state()})

    with pytest.raises(CheckpointCompatibilityError, match="Tokenizer identity mismatch"):
        load_checkpoint(
            mismatched_path,
            CheckpointOperation.MODEL_LOAD,
            runtime_tokenizer_identity=tokenizer_identity,
        )

    selected = discover_checkpoint(
        tmp_path,
        CheckpointOperation.MODEL_LOAD,
        runtime_tokenizer_identity=tokenizer_identity,
    )
    assert selected is not None
    assert selected.path == legacy_path
    assert selected.tokenizer_verified is False

def test_pr4_state_complete_checkpoint_requires_explicit_inexact_opt_in(
    tmp_path, tiny_config, checkpoint_provenance, capsys
):
    checkpoint = _exact_checkpoint(tiny_config, checkpoint_provenance)
    del checkpoint["provenance"]
    path = _save(tmp_path / "pr4-state-complete.pt", checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="provenance-unverified"):
        resolve_resume_checkpoint(
            tmp_path,
            explicit_path=path,
            allow_inexact_resume=False,
            runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
            runtime_data_manifest=checkpoint_provenance["data_manifest"],
        )

    selected, resume_state = resolve_resume_checkpoint(
        tmp_path,
        explicit_path=path,
        allow_inexact_resume=True,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )
    assert selected is not None
    assert resume_state is None
    assert "not exact because" in capsys.readouterr().out
