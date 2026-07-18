from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from byteseed.data_quality import (
    build_data_quality_report,
    create_document,
    data_quality_preprocessing_identity,
    plan_document_dataset,
    validate_data_quality_report,
    write_data_quality_report,
)
from byteseed.eval_prompts import (
    ANCHOR_RETENTION_DEFINITION,
    ANCHOR_RETENTION_PROMPTS,
    CANDIDATE_PARAPHRASE_DEFINITION,
    CANDIDATE_PARAPHRASE_PROMPTS,
)
from byteseed.evaluation import (
    EvaluationValidationError,
    GeneratedCaseOutput,
    GenerationConfig,
    classify_contamination,
    load_evaluation_report,
    logical_checkpoint_identity,
    render_evaluation_report,
    run_evaluation,
    validate_evaluation_report,
    write_evaluation_report,
)
from byteseed.provenance import build_pretraining_data_manifest, canonical_sha256


ENVIRONMENT = {
    "python_version": "test",
    "pytorch_version": "test",
    "device": "cpu",
    "dtype": "fp32",
    "compile": False,
    "deterministic_algorithms_enabled": False,
}


def _generator(cases, _config):
    return [
        GeneratedCaseOutput(
            response="ByteSeed answer",
            generated_token_count=2,
            stop_reason="max_new_tokens",
        )
        for _ in cases
    ]


def _report(*, checkpoint=None, model=None):
    return run_evaluation(
        ANCHOR_RETENTION_DEFINITION,
        GenerationConfig(seed=9),
        _generator,
        checkpoint_identity=checkpoint
        or {"logical_name": "synthetic.pt", "kind": "sft", "version": 1},
        model_configuration=model or {"n_layer": 1, "n_head": 1},
        parameter_count=17,
        tokenizer_identity={"logical_name": "synthetic.model", "sha256": "1" * 64},
        environment=ENVIRONMENT,
    )


def _clean_evidence(tmp_path: Path, tokenizer_identity):
    documents = [
        create_document(
            text=f"unrelated clean document {index}",
            source=f"docs/{index}.md",
            document_id=f"clean-{index}",
        )
        for index in range(10)
    ]
    plan = plan_document_dataset(documents, seed=7, validation_ratio=0.2)
    quality_report = build_data_quality_report(
        plan,
        train_token_count=24,
        validation_token_count=8,
    )
    np.save(tmp_path / "train.npy", np.arange(24, dtype=np.uint16))
    np.save(tmp_path / "val.npy", np.arange(8, dtype=np.uint16))
    write_data_quality_report(tmp_path / "data_quality_report.json", quality_report)
    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.8,
        preprocessing_identity=data_quality_preprocessing_identity(quality_report),
    )
    return quality_report, manifest


def test_report_v1_contains_identity_results_and_deterministic_digest():
    first = _report()
    second = _report()

    assert first["version"] == 1
    assert first["kind"] == "evaluation"
    assert first["suite"]["digest"]
    assert first["checkpoint"]["logical_name"] == "synthetic.pt"
    assert first["parameter_count"] == 17
    assert first["tokenizer_identity"]["logical_name"] == "synthetic.model"
    assert [item["prompt_id"] for item in first["results"]] == [
        case.prompt_id for case in ANCHOR_RETENTION_PROMPTS
    ]
    assert first["digest"] == second["digest"]
    validate_evaluation_report(first)


def test_mapping_insertion_order_does_not_change_report_digest():
    first = _report(
        checkpoint={"logical_name": "synthetic.pt", "kind": "sft", "version": 1},
        model={"n_layer": 1, "n_head": 1},
    )
    second = _report(
        checkpoint={"version": 1, "kind": "sft", "logical_name": "synthetic.pt"},
        model={"n_head": 1, "n_layer": 1},
    )
    assert first["digest"] == second["digest"]


def test_absolute_checkpoint_path_does_not_enter_report_identity(tmp_path):
    identity = logical_checkpoint_identity(
        tmp_path / "private" / "checkpoint.pt",
        version=1,
        kind="sft",
        legacy=False,
        progress=4,
    )
    report = _report(checkpoint=identity)
    encoded = json.dumps(report)
    assert identity["logical_name"] == "checkpoint.pt"
    assert str(tmp_path) not in encoded


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "future evaluation report version"),
        ("kind", "generation_benchmark", "wrong evaluation report kind"),
    ],
)
def test_future_version_and_wrong_kind_fail_clearly(field, value, message):
    report = _report()
    report[field] = value
    with pytest.raises(EvaluationValidationError, match=message):
        validate_evaluation_report(report)


def test_digest_mismatch_and_malformed_result_fail_clearly():
    report = _report()
    report["results"][0]["response"] = "tampered"
    with pytest.raises(EvaluationValidationError, match="digest mismatch"):
        validate_evaluation_report(report)

    malformed = _report()
    del malformed["results"][0]["stop_reason"]
    malformed["digest"] = canonical_sha256(
        {key: value for key, value in malformed.items() if key != "digest"}
    )
    with pytest.raises(EvaluationValidationError, match="stop_reason"):
        validate_evaluation_report(malformed)


def test_duplicate_prompt_id_and_future_rubric_version_fail_clearly():
    duplicate = _report()
    duplicate["results"][1]["prompt_id"] = duplicate["results"][0]["prompt_id"]
    duplicate["digest"] = canonical_sha256(
        {key: value for key, value in duplicate.items() if key != "digest"}
    )
    with pytest.raises(EvaluationValidationError, match="duplicate prompt ID"):
        validate_evaluation_report(duplicate)

    future = _report()
    future["results"][0]["rubric"]["version"] = 2
    future["digest"] = canonical_sha256(
        {key: value for key, value in future.items() if key != "digest"}
    )
    with pytest.raises(EvaluationValidationError, match="rubric version"):
        validate_evaluation_report(future)

    contamination = _report()
    contamination["contamination"]["status"] = "mystery"
    contamination["digest"] = canonical_sha256(
        {key: value for key, value in contamination.items() if key != "digest"}
    )
    with pytest.raises(EvaluationValidationError, match="invalid contamination status"):
        validate_evaluation_report(contamination)


def test_utf8_report_writing_creates_parent_and_does_not_overwrite(tmp_path):
    report = _report()
    destination = tmp_path / "nested" / "evaluation.json"
    write_evaluation_report(destination, report)

    assert destination.read_bytes().decode("utf-8").endswith("\n")
    assert load_evaluation_report(destination) == report
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_report(destination, report)
    write_evaluation_report(destination, report, overwrite=True)


def test_anchor_is_always_rendered_as_contaminated_retention_only():
    report = _report()
    rendered = render_evaluation_report(report)

    assert report["contamination"]["status"] == "contaminated"
    assert report["summary"]["metric_label"] == "Anchor-retention regression"
    assert report["summary"]["held_out_generalization_measured"] is False
    assert "Anchor-retention regression:" in rendered
    assert "Held-out generalization: not yet measured." in rendered
    assert "held-out accuracy" not in rendered.casefold()


def test_candidate_without_audit_is_unverified_and_keeps_all_cases_visible():
    report = run_evaluation(
        CANDIDATE_PARAPHRASE_DEFINITION,
        GenerationConfig(seed=5),
        _generator,
        environment=ENVIRONMENT,
    )
    rendered = render_evaluation_report(report)

    assert report["contamination"]["status"] == "audit_unavailable"
    assert report["contamination"]["held_out_status"] == "unverified"
    assert report["summary"]["total_cases"] == 9
    assert report["summary"]["unscored"] == 1
    assert "Candidate paraphrase checks:" in rendered
    assert "Held-out status: unverified." in rendered


def test_candidate_exact_clean_audit_and_matching_provenance_is_verified(
    tmp_path,
    tokenizer_identity,
):
    quality_report, manifest = _clean_evidence(tmp_path, tokenizer_identity)
    status = classify_contamination(
        CANDIDATE_PARAPHRASE_DEFINITION,
        quality_report=quality_report,
        data_manifest=manifest,
        checkpoint_data_manifest_digest=manifest["digest"],
    )

    assert status["status"] == "verified_clean"
    assert status["held_out_status"] == "verified-clean"
    assert status["suite_covered"] is True

    report = run_evaluation(
        CANDIDATE_PARAPHRASE_DEFINITION,
        GenerationConfig(seed=5),
        _generator,
        checkpoint_identity={
            "logical_name": "synthetic.pt",
            "kind": "pretrain",
            "data_manifest_digest": manifest["digest"],
        },
        data_manifest_digest=manifest["digest"],
        quality_report=quality_report,
        data_manifest=manifest,
        environment=ENVIRONMENT,
    )
    assert report["data_manifest_digest"] == manifest["digest"]
    assert report["contamination"]["quality_report_digest"] == quality_report["digest"]
    assert report["summary"]["metric_label"] == "Candidate held-out paraphrase checks"
    assert report["summary"]["held_out_generalization_measured"] is True


def test_audit_for_other_suite_does_not_verify_candidate(tmp_path):
    documents = [
        create_document(
            text=f"other suite document {index}",
            source=f"docs/{index}.md",
            document_id=f"other-{index}",
        )
        for index in range(10)
    ]
    plan = plan_document_dataset(
        documents,
        seed=3,
        validation_ratio=0.2,
        prompts=ANCHOR_RETENTION_PROMPTS,
    )
    report = build_data_quality_report(plan)
    status = classify_contamination(
        CANDIDATE_PARAPHRASE_DEFINITION,
        quality_report=report,
    )
    assert status["status"] == "audit_does_not_cover_suite"
    assert status["held_out_status"] == "unverified"


def test_pre_pr7_quality_report_without_coverage_remains_valid_but_unverified():
    documents = [
        create_document(
            text=f"legacy report document {index}",
            source=f"docs/{index}.md",
            document_id=f"legacy-{index}",
        )
        for index in range(10)
    ]
    plan = plan_document_dataset(documents, seed=8, validation_ratio=0.2)
    report = build_data_quality_report(plan)
    del report["evaluation_audit"]
    report["digest"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "digest"}
    )

    validate_data_quality_report(report)
    status = classify_contamination(
        CANDIDATE_PARAPHRASE_DEFINITION,
        quality_report=report,
    )
    assert status["status"] == "audit_does_not_cover_suite"


def test_provenance_mismatch_and_contamination_override_remain_visible(
    tmp_path,
    tokenizer_identity,
):
    quality_report, manifest = _clean_evidence(tmp_path, tokenizer_identity)
    mismatch = classify_contamination(
        CANDIDATE_PARAPHRASE_DEFINITION,
        quality_report=quality_report,
        data_manifest=manifest,
        checkpoint_data_manifest_digest="f" * 64,
    )
    assert mismatch["status"] == "provenance_mismatch"

    contaminated_document = create_document(
        text=CANDIDATE_PARAPHRASE_PROMPTS[0].text,
        source="data/sft.jsonl",
        document_id="candidate-overlap",
    )
    clean_document = create_document(
        text="A second unrelated document keeps the synthetic split valid.",
        source="data/clean.md",
        document_id="clean-companion",
    )
    with pytest.warns(UserWarning, match="contamination explicitly accepted"):
        plan = plan_document_dataset(
            [contaminated_document, clean_document],
            seed=4,
            validation_ratio=0.5,
            allow_eval_contamination=True,
        )
    contaminated_report = build_data_quality_report(plan)
    contaminated = classify_contamination(
        CANDIDATE_PARAPHRASE_DEFINITION,
        quality_report=contaminated_report,
    )
    assert contaminated["status"] == "contaminated"
    assert contaminated["override_used"] is True
    assert contaminated["matching_prompt_ids"] == [
        CANDIDATE_PARAPHRASE_PROMPTS[0].prompt_id
    ]
    assert any("override" in warning for warning in contaminated["warnings"])
