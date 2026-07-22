# Generalization SFT v1

generalization-sft-v1 is a data-only ByteSeed v0.5 milestone. It creates a
deterministic supervised fine-tuning source aimed at paraphrase and intent
generalization. It does not train a checkpoint and makes no model-quality claim.

## Baseline and target

The current measured baselines remain:

- **Anchor-retention regression: 9/9.** The Anchor prompts overlap historical
  SFT data, so this is contaminated retention-only evidence.
- **Candidate paraphrase checks: 2/9.** The candidate-paraphrase-v1 suite is
  unverified and remains unchanged.

The weak candidate result shows template retention, concept blending, and poor
transfer to unfamiliar wording. This dataset addresses that training-data gap;
a future checkpoint must still be trained and evaluated before any improvement
can be claimed.

## Dataset composition

The source produces 768 examples: 64 examples in each of 12 concept families.
No family exceeds 8.34% of the dataset.

- ByteSeed identity
- ByteSeed capabilities and limitations
- stack fundamentals
- queue fundamentals
- stack-versus-queue comparison
- overfitting
- underfitting
- overfitting-versus-underfitting contrast
- short DSA study planning
- local ByteSeed workflow
- PyTorch CUDA troubleshooting
- checkpoint and Git hygiene

Each of 96 authored lessons is rendered through eight substantive prompt forms:
direct definition, scenario, comparison, misconception correction, why, how,
action-oriented guidance, and example request. Each form contributes 96
examples. Responses are concise, normally one or two sentences, and avoid
unsupported capability claims.

The source definitions live in
src/byteseed/generalization_sft_source.py. They do not import or transform
evaluation prompt text.

## Deterministic build

From an installed repository environment:

~~~powershell
python scripts/build_generalization_sft.py
~~~

The default outputs are:

- data/raw/generated/generalization_sft_v1.jsonl
- data/raw/generated/generalization_sft_v1.manifest.json
- data/raw/generated/generalization_sft_v1.quality.json

data/raw/generated/ is ignored by Git. Existing outputs are not replaced unless
--overwrite is passed. The --output, --manifest, --quality-report, --source,
and --curated-core options allow explicit paths. Equivalent source bytes and
arguments produce byte-identical JSONL, manifest, and quality report files.

The JSONL uses the current SFT loader's required user and assistant fields. It
also records stable id, dataset, source, category, prompt_form, group_id, and
canonical formatted text fields. UTF-8, LF newlines, key ordering, and record
ordering are deterministic.

## Provenance and quality guards

The SFT manifest and quality report are both schema version 1. Their canonical
SHA-256 identities omit timestamps and absolute paths. The manifest records:

- source and output digests;
- record schema, count, and byte size;
- builder and source versions;
- source-template/semantic-cluster grouping and deterministic split preview;
- the existing curated-core and new generalization components intended for a
  future combined SFT run;
- all audited suite IDs, versions, ordered prompt IDs, and suite digests;
- the linked quality-report digest.

The quality report records family and prompt-form counts, normalized prompt
uniqueness, complete-conversation duplication, response-length distribution,
internal near-wording findings, cross-dataset findings, and exact/near overlap
against every registered evaluation suite. Orphaned or digest-mismatched
artifacts fail validation.

Near-wording policy version 1 uses the existing document normalization as its
base, then compares case-folded word tokens with a fixed 0.82 threshold.
Punctuation, capitalization, whitespace, singular/plural cosmetics, word-order
changes, short prefix/suffix additions, and embedded prompt copies cannot be
used to evade the audit.

Concept-level overlap is intentional: training examples must teach stacks,
queues, fit diagnostics, CUDA checks, and the other supported concepts. Exact
or excessively similar evaluation wording is forbidden.

## Grouping and existing curated data

A stable `group_id` represents one authored lesson/template cluster rather than
an entire concept family. The few merged clusters are reviewed sets of tightly
related operations, such as stack push/pop or queue enqueue/dequeue/front.
Every rendered prompt form for a lesson stays with that cluster, while unrelated
lessons in the same family can be assigned independently.

The 768 records form 93 groups: 91 groups of 8 records, one group of 16, and one
group of 24. Group sizes therefore have minimum 8, median 8, mean 8.258, and
maximum 24. Ten families have eight groups, stack fundamentals has seven, and
queue fundamentals has six.

The canonical seed-20260722 preview is routed through ByteSeed's existing
document-aware splitter. It assigns 648 records to training and 120 to
validation with zero group leakage. Every family appears in both partitions;
training contains 56 records per family except fit contrast, overfitting, and
underfitting (48 each), while validation contains 8 per family except those
same three families (16 each). Repeated builds reproduce group IDs and split
membership exactly.

Whole-family grouping was rejected because it could put all examples for a
required family on one side of the split and because family membership is too
broad to justify a near-duplicate pair.

Every internal near-wording finding is reviewed in the machine-readable quality
report with record IDs, families, group IDs, similarity, classification, and
rationale. A shared group never automatically excuses a finding. The current
source has 67 internal findings: 60 expected same-cluster variants, 7 legitimate
cross-topic stack/queue parallels whose answers clearly distinguish the
structures, and 0 examples requiring rewrite.

generalization-sft-v1 is additive. A future SFT run should combine it with the
tracked 400-record curated personal-assistant core rather than silently
discarding that core. The builder audits exact conversations, normalized
prompts, and near wording across both components and records both file digests
in the intended-training composition. The future checkpoint provenance must
record the exact combined components and tokenizer identity used by that run.

## Generalization holdout

generalization-holdout-v1 contains 24 cases, two for each required concept
family. It uses stable descriptive IDs, deterministic ordering, and transparent
concept rubrics with contradiction checks where concept blending is likely.

The holdout remains excluded from training. It begins as candidate/unverified.
It may be described as held out only after a future checkpoint's exact
tokenizer/data provenance links to a compatible quality audit that covers this
suite version and ordered prompt IDs, reports no overlap, uses no contamination
override, and matches that checkpoint's data identity. Merely registering the
suite or building this dataset does not measure held-out generalization.

## Future workflow

The next model-quality step is deliberately separate:

1. review the generated JSONL and quality report;
2. build the final combined curated-core plus generalization training input;
3. record its tokenizer and data provenance;
4. run SFT under an explicitly approved training task;
5. evaluate Anchor retention, the unchanged candidate suite, and the untouched
   generalization holdout;
6. publish failures as well as passes under the correct contamination labels.

This PR performs none of the training or model evaluation steps.
