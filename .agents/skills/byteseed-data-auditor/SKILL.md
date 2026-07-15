---
name: byteseed-data-auditor
description: Audit ByteSeed training and evaluation data for provenance, licensing, duplication, malformed content, contamination, split integrity, masking, tokenizer reproducibility, and fingerprints. Use when reviewing dataset builders, SFT examples, regression prompts, public data readiness, or data-quality risks.
---

# ByteSeed Data Auditor

Audit repository data and its construction paths without regenerating or modifying artifacts.

## Safety boundary

- Inspect tracked data, examples, builders, docs, metadata, and ignored local metadata read-only.
- Do not run builders, imports, tokenizer training, data preparation, pretraining, SFT, downloads, or commands that rewrite data.
- Do not expose private content. Review secret-pattern matches in context and distinguish detector strings, docs, safety examples, placeholders, and real secrets.

## Audit workflow

1. Map each dataset and example file to its builder, source/provenance field, license note, category, generated/handwritten status, and release policy.
2. Measure exact duplicate user prompts, answers, and pairs. Identify near duplicates, repeated templates/openings, prompt-family imbalance, category leakage, contradictory answers, artificial labels, and malformed text.
3. Compare every evaluation prompt against every SFT source and generated example. Report exact and normalized matches separately; label semantic similarity as a risk unless a defined method demonstrates it.
4. Trace `ChatSFTDataset` prompt/answer encoding, shift, ignored labels, truncation, padding, and the possibility of all-ignored targets.
5. Trace pretraining corpus assembly and token splitting. Identify document-fragment boundaries, train/validation adjacency, repeated-document leakage, and whether split/fingerprint manifests exist.
6. Compare configured and effective vocabulary, special-token metadata, tokenizer/corpus hashes, checkpoint fingerprints, and public artifact availability.

## Evidence rules

Classify items as `Strength`, `Confirmed defect`, `Suspected risk`, or `Missing test`. Do not turn near-duplicate or leakage concerns into confirmed defects without a demonstrated match or broken invariant.

For every defect, risk, and missing test include `path:start-end`, concrete evidence, user-visible impact, the smallest safe correction, validation, and compatibility implications.

## Expected output

Report provenance and licensing coverage, duplicate/near-duplicate evidence, malformed and template-dominant examples, category and prompt overlap, masking/truncation findings, split integrity, tokenizer/corpus reproducibility, missing fingerprints, tests needed, and ranked corrective actions. Keep anchor retention separate from held-out generalization.
