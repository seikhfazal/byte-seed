"""Build generalization-sft-v1 JSONL and linked provenance artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from byteseed.generalization_sft import write_generalization_artifacts
from byteseed.generalization_sft_source import DATASET_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "generated" / "generalization_sft_v1.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "generated" / "generalization_sft_v1.manifest.json"
DEFAULT_QUALITY = ROOT / "data" / "raw" / "generated" / "generalization_sft_v1.quality.json"
DEFAULT_SOURCE = ROOT / "src" / "byteseed" / "generalization_sft_source.py"
DEFAULT_CURATED_CORE = (
    ROOT / "data" / "raw" / "assistant_sft" / "curated_personal_assistant_core.jsonl"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic generalization-sft-v1 data-only artifacts."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--curated-core", type=Path, default=DEFAULT_CURATED_CORE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files; without this flag the builder fails closed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, quality = write_generalization_artifacts(
        output_path=args.output,
        manifest_path=args.manifest,
        quality_report_path=args.quality_report,
        source_path=args.source,
        curated_core_path=args.curated_core,
        overwrite=args.overwrite,
    )
    print(f"Dataset: {DATASET_VERSION}")
    print(f"Examples: {quality['counts']['records']}")
    for family, count in quality["family_counts"].items():
        print(f"  {family}: {count}")
    print(f"JSONL: {args.output}")
    print(f"Manifest: {args.manifest}")
    print(f"Quality report: {args.quality_report}")
    print(f"Exact duplicates: {quality['counts']['exact_duplicate_prompts']}")
    print(f"Internal near duplicates: {quality['deduplication']['internal_near_count']}")
    print(
        "Near review: "
        f"{quality['deduplication']['review_summary']['expected same-cluster variant']} "
        "expected cluster variants, "
        f"{quality['deduplication']['review_summary']['legitimate cross-topic wording']} "
        "legitimate cross-topic, "
        f"{quality['deduplication']['review_summary']['rewrite required']} rewrites required"
    )
    print(
        "Split preview: "
        f"{quality['split_readiness']['record_counts']['train']} train / "
        f"{quality['split_readiness']['record_counts']['validation']} validation "
        f"across {len(quality['split_readiness']['groups'])} groups"
    )
    print(f"Evaluation exact overlap: {len(quality['evaluation_audit']['exact_findings'])}")
    print(f"Evaluation near overlap: {len(quality['evaluation_audit']['near_findings'])}")
    print(f"Manifest digest: {manifest['digest']}")
    print(f"Quality-report digest: {quality['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
