from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from byteseed.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointKind,
    build_checkpoint,
    build_resume_state,
    training_config_snapshot,
    validate_exact_resume_checkpoint,
)
from byteseed.data_quality import (
    build_data_quality_report,
    create_document,
    data_quality_preprocessing_identity,
    plan_document_dataset,
    write_data_quality_report,
)
from byteseed.provenance import (
    DATA_MANIFEST_VERSION,
    LEGACY_DATA_MANIFEST_VERSION,
    ProvenanceValidationError,
    build_checkpoint_provenance,
    build_pretraining_data_manifest,
    canonical_sha256,
    create_data_manifest,
    validate_data_manifest,
    write_data_manifest,
)


def _documents():
    return [
        create_document(
            text=f"manifest document {index}",
            source=f"docs/{index}.md",
            document_id=f"manifest-{index}",
        )
        for index in range(10)
    ]


def _report(*, seed=3, ratio=0.2):
    plan = plan_document_dataset(
        _documents(),
        seed=seed,
        validation_ratio=ratio,
        prompts=(),
    )
    return build_data_quality_report(
        plan, train_token_count=24, validation_token_count=8
    )


def _arrays(directory):
    np.save(directory / "train.npy", np.arange(24, dtype=np.uint16))
    np.save(directory / "val.npy", np.arange(8, dtype=np.uint16))


def test_document_aware_manifest_v2_records_quality_and_split_policy(
    tmp_path, tokenizer_identity
):
    _arrays(tmp_path)
    report = _report()
    write_data_quality_report(tmp_path / "data_quality_report.json", report)

    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(report),
    )

    assert manifest["version"] == DATA_MANIFEST_VERSION == 2
    assert manifest["preprocessing"]["tokenization"]["per_document"] is True
    assert manifest["preprocessing"]["split"]["seed"] == 3
    assert (
        manifest["preprocessing"]["data_quality"]["report_digest"]
        == report["digest"]
    )
    validate_data_manifest(manifest)


def test_persisted_v2_manifest_and_report_are_reused_for_runtime_identity(
    tmp_path, tokenizer_identity
):
    _arrays(tmp_path)
    report = _report()
    write_data_quality_report(tmp_path / "data_quality_report.json", report)
    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(report),
    )
    write_data_manifest(tmp_path / "data_manifest.json", manifest)

    runtime = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
    )

    assert runtime == manifest


def test_missing_quality_report_blocks_runtime_v2_manifest_reconstruction(
    tmp_path, tokenizer_identity
):
    _arrays(tmp_path)
    report = _report()
    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(report),
    )
    write_data_manifest(tmp_path / "data_manifest.json", manifest)

    with pytest.raises(FileNotFoundError, match="data-quality report is missing"):
        build_pretraining_data_manifest(
            tmp_path,
            tokenizer_identity=tokenizer_identity,
            train_split=0.8,
        )



def test_orphaned_quality_report_blocks_legacy_manifest_fallback(
    tmp_path, tokenizer_identity
):
    _arrays(tmp_path)
    write_data_quality_report(tmp_path / "data_quality_report.json", _report())

    with pytest.raises(ProvenanceValidationError) as exc_info:
        build_pretraining_data_manifest(
            tmp_path,
            tokenizer_identity=tokenizer_identity,
            train_split=0.8,
        )

    message = str(exc_info.value).lower()
    assert "orphaned data-quality report" in message
    assert "document-aware v2 manifest is missing" in message

def test_split_seed_quality_policy_and_override_change_manifest_digest(
    tmp_path, tokenizer_identity
):
    _arrays(tmp_path)
    base_report = _report(seed=1)
    seed_report = _report(seed=2)
    dedup_report = copy.deepcopy(base_report)
    dedup_report["deduplication"]["representative_order"] = "source,document-id"
    dedup_report["digest"] = canonical_sha256(
        {key: value for key, value in dedup_report.items() if key != "digest"}
    )
    override_report = copy.deepcopy(base_report)
    override_report["policy"]["allow_eval_contamination"] = True
    override_report["digest"] = canonical_sha256(
        {key: value for key, value in override_report.items() if key != "digest"}
    )

    manifests = [
        build_pretraining_data_manifest(
            tmp_path,
            tokenizer_identity=tokenizer_identity,
            train_split=0.8,
            preprocessing_identity=data_quality_preprocessing_identity(report),
        )
        for report in (base_report, seed_report, dedup_report, override_report)
    ]

    assert len({manifest["digest"] for manifest in manifests}) == 4


def test_legacy_manifest_v1_remains_valid_under_original_semantics(data_manifest):
    assert data_manifest["version"] == LEGACY_DATA_MANIFEST_VERSION

    validate_data_manifest(data_manifest)


def test_manifest_v2_does_not_reinterpret_v1_preprocessing(
    data_manifest,
):
    with pytest.raises(ProvenanceValidationError, match="versions must match"):
        create_data_manifest(
            tokenizer_identity=data_manifest["tokenizer"],
            artifacts=data_manifest["artifacts"],
            preprocessing=data_manifest["preprocessing"],
            manifest_version=DATA_MANIFEST_VERSION,
        )


def test_future_manifest_version_fails_without_breaking_v1(data_manifest):
    future = dict(data_manifest, version=DATA_MANIFEST_VERSION + 1)

    with pytest.raises(ProvenanceValidationError, match="Unsupported data manifest"):
        validate_data_manifest(future)
    validate_data_manifest(data_manifest)


def test_exact_resume_accepts_matching_v2_and_rejects_changed_split_policy(
    tmp_path, tokenizer_identity, tiny_config
):
    _arrays(tmp_path)
    report = _report(seed=7)
    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(report),
    )
    critical = training_config_snapshot(
        tiny_config.__dict__, device_type="cpu", amp_enabled=False
    )
    resume_state = build_resume_state(
        scaler=None,
        best_val=0.5,
        patience_left=2,
        training_config=critical,
    )
    checkpoint = build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"state": {}, "param_groups": []},
        config=tiny_config.__dict__,
        iteration=3,
        best_val=0.5,
        resume_state=resume_state,
        provenance=build_checkpoint_provenance(
            tokenizer_identity, data_manifest=manifest
        ),
    )

    validate_exact_resume_checkpoint(
        checkpoint,
        runtime_tokenizer_identity=tokenizer_identity,
        runtime_data_manifest=manifest,
    )

    changed_report = _report(seed=8)
    changed_manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(changed_report),
    )
    with pytest.raises(CheckpointCompatibilityError, match="split configuration"):
        validate_exact_resume_checkpoint(
            checkpoint,
            runtime_tokenizer_identity=tokenizer_identity,
            runtime_data_manifest=changed_manifest,
        )
