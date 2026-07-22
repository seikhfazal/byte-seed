# ByteSeed Data Quality

PR 6 makes new pretraining-data builds document-aware and records the policy that produced them. It does not rewrite historical datasets.

## Supported document sources

The document-aware reader uses real source-record boundaries:

- top-level `data/raw/*.md` files, except the historical combined `byteseed_personal_assistant_corpus.md`;
- `data/raw/personal_assistant/*.md`;
- `data/raw/generated/markdown/*.md`, when locally present;
- top-level JSONL records with either a `text` field or non-empty `user` and `assistant` fields.

Each Markdown file and each JSONL record is one document. Arbitrary token chunks are never presented as document boundaries. The old combined corpus and contiguous token split remain available only as historical manifest-v1 identity; `read_legacy_markdown_corpus` is the explicitly named reproduction helper.

An explicit JSONL `id` is preserved. Without one, ByteSeed derives an ID from the normalized logical source and canonical record content. IDs and fingerprints never include absolute machine paths or input positions.

## Normalization and duplicate policy

Normalization version `1`:

- applies Unicode NFC;
- normalizes CRLF and CR newlines to LF;
- removes outer blank space;
- collapses ordinary spaces and tabs on prose lines;
- preserves indentation on indented lines and spacing inside fenced code.

Every record has a raw-content SHA-256 and a canonical-text SHA-256. Raw duplicates and additional whitespace/Unicode-equivalent canonical duplicates are reported separately. A canonical duplicate group is assigned as one unit, so copies cannot cross train and validation.

New preparation retains one representative per duplicate group. The representative is selected by stable document-ID, logical-source, and raw-fingerprint ordering. Removed IDs and sources are recorded in the report. Conflicting explicit split assignments inside one group are errors.

## Deterministic splitting and leakage checks

Split-strategy version `1` hashes the canonical duplicate-group fingerprint together with the configured seed. The fixed hash value is compared with `1 - train_split`; input order and filesystem enumeration order do not affect assignment.

At least two unique duplicate groups are required. If a tiny valid collection hashes entirely to one side, a deterministic score-based fallback keeps both splits non-empty. Explicit assignments that make a required split empty fail.

Before writing arrays, ByteSeed rejects cross-split reuse of:

- a document ID;
- a raw fingerprint;
- a canonical fingerprint or duplicate group.

Each split is then tokenized independently. Every document receives its own existing BOS/EOS boundary; no new tokenizer symbol is introduced.

## Evaluation-contamination policy

The versioned evaluation registry is shared by the stable evaluation runner and contamination checker. It contains the unchanged historical Anchor retention suite and the candidate paraphrase suite; the generation benchmark continues to use an Anchor prompt as its default input. Matching uses normalization version `1` and detects:

- canonical full-document equality;
- exact structured-field equality;
- exact normalized substring occurrence.

This is deliberately exact and conservative. PR 6 does not add semantic, fuzzy, embedding, MinHash, or near-duplicate matching.

New preparation fails by default when a registered prompt is found. Historical reproduction requires the single explicit override:

```powershell
python -m src.byteseed.prepare_data --config configs/byteseed_12m.yaml --allow-eval-contamination
```

The override emits a warning and is recorded in both report and manifest identity. It never turns a contaminated result into held-out evidence. `audit_sft_file` provides a non-writing report-only path for existing SFT JSONL files.

Historical status remains:

- Anchor-retention regression: 9/9.
- Held-out generalization: not yet measured.

All nine stable Anchor prompts occur verbatim in the tracked Anchor v2.3 SFT material. PR 6 detects future overlap; it does not retroactively cleanse or rewrite that historical file.

## Report and provenance

`data_quality_report.json` version `1` is canonical JSON and records:

- normalization, document-format, deduplication, and split versions;
- split seed and validation ratio;
- input, unique, duplicate, removed, train, validation, token, and contamination counts;
- removed duplicate IDs/sources;
- contamination findings with prompt ID, suite, split, document ID, logical source, match type, and shortened fingerprint;
- evaluation-registry version plus each audited suite version and exact ordered prompt IDs;
- leakage-validation result and contamination override status;
- a deterministic SHA-256 report digest.

It omits timestamps, absolute paths, and document bodies. Equivalent inputs in different order produce the same digest. Earlier report-v1 files without the additive exact-coverage field remain valid, but they cannot verify that a zero finding count covers the candidate suite. See [EVALUATION.md](EVALUATION.md) for the full held-out classification rules.

Document-aware builds use data-manifest version `2`. The manifest continues to fingerprint `train.npy`, `val.npy`, and tokenizer identity, and additionally includes the report digest plus document, normalization, deduplication, split, seed, ratio, and contamination policy. Changing any of those identity fields changes the manifest digest.

Manifest version `1` remains valid under its original contiguous-token semantics. It is never reinterpreted as document-aware. Exact resume compares the originally stored manifest version and digest; a v2 runtime also requires the persisted quality report to match its manifest. Inference and legacy Anchor checkpoint compatibility are unchanged.

## Generalization SFT data

generalization-sft-v1 uses a focused SFT manifest and quality-report schema,
both version 1, without changing the pretraining manifest contracts above. The
SFT report reuses the registered exact-contamination detector and adds
near-wording policy version 1 at a fixed 0.82 threshold. The comparison
case-folds and tokenizes the existing normalized document text, then considers
sequence order, token overlap, multiset overlap, and contained prompt wording.

The SFT audit fails when any registered suite has an exact or near finding.
It records all three suite versions, ordered IDs, and canonical suite digests,
plus internal and curated-core overlap findings. A report without its versioned
manifest is orphaned and invalid; a manifest without its report, a link digest
mismatch, or an output JSONL digest mismatch also fails closed.

Its split-readiness section assigns stable source-template/semantic clusters,
then routes those groups through the existing document-aware splitter. The
report records partition and per-family counts, group-size distribution, split
membership, leakage status, and a canonical split-preview digest. Validation
requires every required family in both train and validation and rejects any
group shared across partitions.

Every internal near finding is retained even when records share a group. The
review section records both record IDs, concept families, group IDs, similarity,
and one of three classifications: expected same-cluster variant, legitimate
cross-topic wording, or rewrite required. A build fails while any finding still
requires rewrite; same-group membership alone is not an exception.

Concept overlap is expected and allowed. Exact or cosmetically rewritten
evaluation wording is not. See [Generalization SFT v1](GENERALIZATION_SFT_V1.md)
for the source, builder, grouping, and future-training policy.
