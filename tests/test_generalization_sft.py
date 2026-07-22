from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from byteseed.data_quality import (
    NEAR_DUPLICATE_THRESHOLD,
    NearDuplicateFinding,
    detect_near_duplicate_texts,
    near_duplicate_similarity,
)
from byteseed.finetune_chat import ChatSFTDataset, IGNORE_INDEX
from byteseed.generalization_sft import (
    GeneralizationSFTValidationError,
    assign_group_splits,
    build_generalization_artifacts,
    build_generalization_records,
    build_split_readiness,
    jsonl_bytes,
    review_internal_near_findings,
    validate_generalization_records,
    validate_sft_artifact_files,
    validate_sft_manifest,
    validate_sft_quality_report,
    write_generalization_artifacts,
)
from byteseed.generalization_sft_source import (
    DATASET_VERSION,
    PROMPT_FORMS,
    REQUIRED_FAMILIES,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "byteseed" / "generalization_sft_source.py"
CURATED_CORE = (
    ROOT / "data" / "raw" / "assistant_sft" / "curated_personal_assistant_core.jsonl"
)


class CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, *, add_bos=False, add_eos=False):
        ids = list(range(10, 10 + len(text)))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids


@pytest.fixture(scope="module")
def records():
    return build_generalization_records()


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("generalization-artifacts")
    output = directory / "data.jsonl"
    manifest = directory / "manifest.json"
    quality = directory / "quality.json"
    written_manifest, written_quality = write_generalization_artifacts(
        output_path=output,
        manifest_path=manifest,
        quality_report_path=quality,
        source_path=SOURCE,
        curated_core_path=CURATED_CORE,
    )
    return directory, output, manifest, quality, written_manifest, written_quality


def test_dataset_count_balance_forms_and_prompt_uniqueness(records):
    assert len(records) == 768
    assert Counter(row["category"] for row in records) == {
        family: 64 for family in REQUIRED_FAMILIES
    }
    assert Counter(row["prompt_form"] for row in records) == {
        form: 96 for form in PROMPT_FORMS
    }
    assert len({row["user"].casefold() for row in records}) == 768
    assert max(Counter(row["category"] for row in records).values()) / len(records) < 0.15


def test_output_order_and_jsonl_are_deterministic(records):
    assert records == build_generalization_records()
    assert jsonl_bytes(records) == jsonl_bytes(build_generalization_records())
    assert jsonl_bytes(records).endswith(b"\n")
    assert not jsonl_bytes(records).startswith(b"\xef\xbb\xbf")


def test_manifest_and_quality_report_are_byte_deterministic(tmp_path):
    kwargs = {
        "source_path": SOURCE,
        "curated_core_path": CURATED_CORE,
    }
    first = build_generalization_artifacts(
        output_path=tmp_path / "one" / "data.jsonl",
        manifest_path=tmp_path / "one" / "manifest.json",
        quality_report_path=tmp_path / "one" / "quality.json",
        **kwargs,
    )
    second = build_generalization_artifacts(
        output_path=tmp_path / "two" / "data.jsonl",
        manifest_path=tmp_path / "two" / "manifest.json",
        quality_report_path=tmp_path / "two" / "quality.json",
        **kwargs,
    )
    assert first == second


def test_artifacts_validate_and_link_to_the_output(built_artifacts):
    _, output, manifest_path, quality_path, manifest, quality = built_artifacts
    validate_sft_quality_report(quality)
    validate_sft_manifest(manifest, quality_report=quality)
    loaded_manifest, loaded_quality = validate_sft_artifact_files(
        output_path=output,
        manifest_path=manifest_path,
        quality_report_path=quality_path,
    )
    assert loaded_manifest == manifest
    assert loaded_quality == quality
    assert manifest["quality_report"]["digest"] == quality["digest"]
    assert manifest["grouping"]["strategy"] == "source-template-semantic-cluster"
    assert manifest["grouping"]["group_count"] == 93
    assert manifest["grouping"]["group_size_distribution"] == {
        "minimum": 8,
        "median": 8.0,
        "mean": 8.258,
        "maximum": 24,
    }
    assert manifest["build_configuration"]["randomness"] == "none"
    assert [item["name"] for item in manifest["intended_training_components"]] == [
        "curated-personal-assistant-core",
        DATASET_VERSION,
    ]
    assert str(output.parent) not in json.dumps(manifest)
    assert str(output.parent) not in json.dumps(quality)


def test_overwrite_protection_and_explicit_overwrite(built_artifacts):
    _, output, manifest, quality, _, _ = built_artifacts
    kwargs = {
        "output_path": output,
        "manifest_path": manifest,
        "quality_report_path": quality,
        "source_path": SOURCE,
        "curated_core_path": CURATED_CORE,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_generalization_artifacts(**kwargs)
    write_generalization_artifacts(**kwargs, overwrite=True)


def test_orphaned_quality_report_fails_closed(tmp_path):
    quality = tmp_path / "quality.json"
    quality.write_text("{}", encoding="utf-8")
    with pytest.raises(
        GeneralizationSFTValidationError,
        match="orphaned SFT data-quality report.*manifest is missing",
    ):
        validate_sft_artifact_files(
            output_path=tmp_path / "data.jsonl",
            manifest_path=tmp_path / "manifest.json",
            quality_report_path=quality,
        )


def test_manifest_digest_and_suite_digest_tampering_fail(built_artifacts):
    _, _, _, _, manifest, quality = built_artifacts
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["output"]["record_count"] += 1
    with pytest.raises(GeneralizationSFTValidationError, match="manifest digest mismatch"):
        validate_sft_manifest(bad_manifest, quality_report=quality)

    bad_quality = copy.deepcopy(quality)
    bad_quality["evaluation_audit"]["suites"][0]["suite_digest"] = "0" * 64
    with pytest.raises(GeneralizationSFTValidationError, match="suite digest mismatch"):
        validate_sft_quality_report(bad_quality)

    incomplete = copy.deepcopy(quality)
    incomplete["evaluation_audit"]["suites"][-1]["prompt_ids"].pop()
    with pytest.raises(GeneralizationSFTValidationError, match="coverage is incomplete"):
        validate_sft_quality_report(incomplete)


def test_quality_report_is_clean_against_all_suites_and_curated_core(built_artifacts):
    *_, quality = built_artifacts
    audit = quality["evaluation_audit"]
    assert [entry["suite_id"] for entry in audit["suites"]] == [
        "anchor-retention-v0.2",
        "candidate-paraphrase-v1",
        "generalization-holdout-v1",
    ]
    assert audit["exact_findings"] == []
    assert audit["near_findings"] == []
    assert quality["cross_dataset"]["exact_prompt_overlap_count"] == 0
    assert quality["cross_dataset"]["exact_conversation_overlap_count"] == 0
    assert quality["cross_dataset"]["near_prompt_overlap_count"] == 0
    assert (
        quality["deduplication"]["within_group_near_count"]
        + quality["deduplication"]["cross_group_near_count"]
        == quality["deduplication"]["internal_near_count"]
    )
    assert quality["deduplication"]["review_summary"]["rewrite required"] == 0
    assert len(quality["deduplication"]["internal_near_review"]) == (
        quality["deduplication"]["internal_near_count"]
    )
    cross_family = [
        item
        for item in quality["deduplication"]["internal_near_review"]
        if item["left_family"] != item["right_family"]
    ]
    assert len(cross_family) == 7
    assert {
        item["classification"] for item in cross_family
    } == {"legitimate cross-topic wording"}


def test_current_sft_loader_accepts_every_record_with_supervision(tmp_path, records):
    path = tmp_path / "data.jsonl"
    path.write_bytes(jsonl_bytes(records))
    dataset = ChatSFTDataset(path, CharacterTokenizer(), block_size=256, device="cpu")
    assert len(dataset.examples) == 768
    assert all(any(label != IGNORE_INDEX for label in labels) for _, labels in dataset.examples)


def test_canonical_document_split_has_family_coverage_and_zero_group_leakage(records):
    assignments = assign_group_splits(records)
    preview = build_split_readiness(records)
    assert set(assignments.values()) == {"train", "validation"}
    assert preview["record_counts"] == {"train": 648, "validation": 120}
    assert preview["group_leakage"] == []
    assert preview["group_counts"] == {
        "byteseed_capabilities_limitations": 8,
        "byteseed_identity": 8,
        "checkpoint_git_hygiene": 8,
        "cuda_troubleshooting": 8,
        "dsa_study_planning": 8,
        "fit_contrast": 8,
        "local_workflow": 8,
        "overfitting": 8,
        "queue_fundamentals": 6,
        "stack_fundamentals": 7,
        "stack_queue_comparison": 8,
        "underfitting": 8,
    }
    for family in REQUIRED_FAMILIES:
        group_ids = {row["group_id"] for row in records if row["category"] == family}
        assert len(group_ids) >= 6
        assert {assignments[group_id] for group_id in group_ids} == {
            "train",
            "validation",
        }
        assert preview["family_counts"]["train"][family] > 0
        assert preview["family_counts"]["validation"][family] > 0
    for group_id in assignments:
        split_memberships = {
            entry["split"]
            for entry in preview["groups"]
            if entry["group_id"] == group_id
        }
        assert split_memberships == {assignments[group_id]}


def test_related_prompt_forms_and_operation_pairs_remain_clustered(records):
    by_source = {}
    for record in records:
        by_source.setdefault(record["source"], set()).add(record["group_id"])
    assert len(by_source) == 96
    assert all(len(group_ids) == 1 for group_ids in by_source.values())
    assert {
        row["group_id"]
        for row in records
        if row["source"].endswith(("stack-push", "stack-pop"))
    } == {"generalization-sft-v1:stack_fundamentals:stack-push-pop"}
    assert {
        row["group_id"]
        for row in records
        if row["source"].endswith(("queue-enqueue", "queue-dequeue", "queue-front"))
    } == {"generalization-sft-v1:queue_fundamentals:queue-operations"}


def test_group_ids_and_split_membership_repeat_identically(records):
    rebuilt = build_generalization_records()
    assert [row["group_id"] for row in records] == [row["group_id"] for row in rebuilt]
    assert assign_group_splits(records) == assign_group_splits(rebuilt)
    assert build_split_readiness(records) == build_split_readiness(rebuilt)


def test_same_group_does_not_excuse_an_unreviewed_near_duplicate():
    records = (
        {
            "id": "left",
            "category": "overfitting",
            "group_id": "synthetic-shared-group",
            "source": "lesson-left",
        },
        {
            "id": "right",
            "category": "overfitting",
            "group_id": "synthetic-shared-group",
            "source": "lesson-right",
        },
    )
    review = review_internal_near_findings(
        records,
        (NearDuplicateFinding("left", "right", 0.99),),
    )
    assert review[0]["classification"] == "rewrite required"


def test_unreviewed_stack_queue_pair_is_not_excused_by_topic_relationship():
    records = (
        {
            "id": "gen-v1.queue-new.direct_definition",
            "category": "queue_fundamentals",
            "group_id": "queue-new",
            "source": "queue-new",
        },
        {
            "id": "gen-v1.stack-new.direct_definition",
            "category": "stack_fundamentals",
            "group_id": "stack-new",
            "source": "stack-new",
        },
    )
    review = review_internal_near_findings(
        records,
        (
            NearDuplicateFinding(
                "gen-v1.queue-new.direct_definition",
                "gen-v1.stack-new.direct_definition",
                0.99,
            ),
        ),
    )
    assert review[0]["classification"] == "rewrite required"


def test_source_generation_does_not_depend_on_evaluation_registry():
    source = SOURCE.read_text(encoding="utf-8")
    assert "eval_prompts" not in source
    assert "registered_evaluation" not in source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user", " ", "non-empty"),
        ("assistant", "TODO", "placeholder"),
        ("assistant", "Run python chat.py.py.", "malformed chat command"),
        ("assistant", "Call torch.cuda.isavailable().", "malformed CUDA API"),
        ("category", "unknown-family", "invalid concept family"),
    ],
)
def test_narrow_content_and_schema_guards_reject_bad_records(records, field, value, message):
    changed = [dict(row) for row in records]
    changed[0][field] = value
    if field in {"user", "assistant"}:
        changed[0]["text"] = (
            f"<|user|>\n{changed[0]['user']}\n<|assistant|>\n"
            f"{changed[0]['assistant']}\n<|end|>"
        )
    with pytest.raises(GeneralizationSFTValidationError, match=message):
        validate_generalization_records(changed)


def test_duplicate_prompt_and_conversation_guards(records):
    changed = [dict(row) for row in records]
    changed[1]["user"] = changed[0]["user"]
    changed[1]["assistant"] = changed[0]["assistant"]
    changed[1]["text"] = changed[0]["text"]
    with pytest.raises(GeneralizationSFTValidationError, match="duplicate normalized user prompt"):
        validate_generalization_records(changed)


def test_fit_contradiction_guard_rejects_wrong_family_answer(records):
    changed = [dict(row) for row in records]
    index = next(
        index
        for index, row in enumerate(changed)
        if row["category"] == "overfitting"
    )
    changed[index]["assistant"] = "This is underfitting."
    changed[index]["text"] = (
        f"<|user|>\n{changed[index]['user']}\n<|assistant|>\n"
        f"{changed[index]['assistant']}\n<|end|>"
    )
    with pytest.raises(GeneralizationSFTValidationError, match="contradicts the overfitting"):
        validate_generalization_records(changed)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Which item leaves this queue first?", "which item leaves this queue first"),
        ("WHICH ITEM LEAVES THIS QUEUE FIRST", "Which item leaves this queue first?"),
        ("Which items leave these queues first?", "Which item leaves this queue first?"),
        (
            "Explain why queued requests are processed in earliest to latest order.",
            "Queued requests are processed in earliest to latest order; explain why.",
        ),
        (
            "Please answer: Which item leaves this queue first? Keep it short.",
            "Which item leaves this queue first?",
        ),
        (
            "Context for a beginner: Which item leaves this queue first? Then give one reason.",
            "Which item leaves this queue first?",
        ),
    ],
)
def test_cosmetic_and_embedded_prompt_variants_are_near_duplicates(left, right):
    assert near_duplicate_similarity(left, right) >= NEAR_DUPLICATE_THRESHOLD


def test_legitimate_shared_topic_scenarios_are_not_near_duplicates():
    first = "A printer receives jobs A, B, and C. Which ordinary job runs next?"
    second = "Use a stack to trace three nested function calls and explain the return order."
    assert detect_near_duplicate_texts((("printer", first),), (("calls", second),)) == ()


def test_generated_jsonl_has_current_canonical_fields(built_artifacts):
    _, output, *_ = built_artifacts
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 768
    assert set(rows[0]) == {
        "id", "dataset", "source", "category", "prompt_form", "group_id",
        "user", "assistant", "text",
    }
    assert {row["dataset"] for row in rows} == {DATASET_VERSION}
