# Evaluation and Generation Benchmarking

ByteSeed evaluation is separate from training. Evaluation scripts load an existing local checkpoint and tokenizer, generate responses, apply transparent deterministic checks, and optionally write versioned JSON reports. They do not update model weights or save checkpoints.

The current published result remains:

- Anchor-retention regression: 9/9.
- Held-out generalization: not yet measured.

The Anchor result is a historical retention regression with known training overlap. It is not a held-out metric, evidence of generalization, or a broad benchmark score.

## Evaluation suites

| Suite ID | Suite version | Purpose | Initial status | Cases |
| --- | ---: | --- | --- | ---: |
| `anchor-retention-v0.2` | 1 | Historical retention regression | Known contaminated | 9 |
| `candidate-paraphrase-v1` | 1 | Candidate generalization checks | Candidate/unverified | 9 |

The Anchor prompt text, IDs, and ordering remain unchanged. All nine prompts occur verbatim in historical Anchor v2.3 SFT material, so an Anchor report is always classified as contaminated and retention-only.

The candidate suite has one materially reworded case for each Anchor concept. Stable IDs are independent of list position. Candidate wording is not itself evidence of held-out status: the suite starts unverified and becomes verified clean only for a specific audited data identity.

Both suites are part of the shared evaluation-prompt registry. New data-quality reports record registry version, suite version, and the exact ordered prompt IDs that were audited. Older data-quality report-v1 files remain valid, but a report that does not record exact suite coverage cannot verify a candidate suite as clean.

## Deterministic rubrics

Rubric version 1 uses normalized response concepts rather than model-generated grading. Normalization applies Unicode NFKC, case folding, and whitespace collapse. A required concept may define several accepted phrases. Forbidden terms or combinations are applied only when the case declares them.

Each case result records matched and missing required concepts, forbidden matches, and one of these statuses:

- `pass`: every required concept is present and no declared contradiction is present.
- `fail`: a required concept is missing, the response is empty, or a declared contradiction is present.
- `human-review`: deterministic concepts are reported, but the case is not automatically passed.
- `unscored`: no deterministic rubric exists.

Failed, human-review, and unscored cases remain visible in the suite denominator. They are never silently counted as passing.

## Deterministic generation boundary

Every shared-runner invocation records an explicit seed, temperature, `top_k`, maximum generated-token count, repetition penalty, stop-token IDs, stop-at-end policy, device, dtype, compile status, batch size, prompt-format version, sampling mode, and deterministic-algorithm setting.

The stable evaluation defaults remain seed `1337`, temperature `0.2`, `top_k=5`, `max_new_tokens=80`, and repetition penalty `1.0`. Stochastic evaluation seeds Python, the Torch CPU generator, and CUDA generators only when CUDA evaluation is requested and CUDA is already initialized. The caller's prior RNG and deterministic-algorithm state is restored in a `finally` path, including when generation fails. `top_k=1` with greedy mode is deterministic by construction; its generated text does not depend on the sampling seed.

Equivalent inputs are expected to reproduce under the same supported software, hardware, device, dtype, and deterministic-kernel conditions. ByteSeed does not promise bitwise-identical output across CPU and CUDA, different GPUs, different PyTorch/CUDA versions, or nondeterministic kernels.

## Running evaluation

The historical invocation remains valid:

```powershell
python scripts/eval_stable_v0_2.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

To write an Anchor report without silently replacing an existing file:

```powershell
python scripts/eval_stable_v0_2.py `
  --checkpoint checkpoints\anchor_v2_3_finetuned.pt `
  --seed 1337 `
  --output-json runs\evaluation\anchor-retention-v0.2.json
```

Run the candidate suite with explicit PR 6 evidence when the selected checkpoint carries matching data provenance:

```powershell
python scripts/eval_stable_v0_2.py `
  --suite candidate-paraphrase-v1 `
  --checkpoint checkpoints\candidate.pt `
  --seed 1337 `
  --data-quality-report data\processed\data_quality_report.json `
  --data-manifest data\processed\data_manifest.json `
  --output-json runs\evaluation\candidate-paraphrase-v1.json
```

The output parent is created when needed. Existing JSON files are rejected unless `--overwrite` is explicitly supplied. An unknown suite, malformed configuration, invalid evidence file, or invalid report request exits nonzero. Human-readable output remains available whether or not JSON output is requested.

Historical `eval_anchor_v2*.py` and `eval_chat_checkpoint.py` entry points remain available for their established diagnostics. They now accept `--seed`; the stable shared runner above is the report-producing interface.

## Evaluation report version 1

An evaluation report contains:

- report kind/version and runner/prompt-format versions;
- exact suite ID, version, purpose, case count, and suite digest;
- the complete generation configuration and seed;
- path-safe checkpoint identity and kind/version/progress when available;
- model configuration summary and parameter count;
- verified tokenizer identity when available;
- checkpoint data-manifest digest when available;
- contamination classification, exact matching prompt IDs, override status, and warnings;
- Python/PyTorch runtime versions plus device, dtype, compile, and deterministic-algorithm status;
- ordered prompt results with prompt digest, response, stop reason, generated-token count, rubric result, and warnings;
- transparent passed/failed/unscored aggregates; and
- a canonical report digest.

Reports do not contain absolute local paths, checkpoint tensors, training data, timestamps, memory addresses, or unstable object representations. The report digest is SHA-256 over compact UTF-8 canonical JSON with sorted keys and the `digest` field omitted. Prompt and result ordering follows the suite definition. Equivalent reports therefore have the same digest; changing a response, setting, identity, classification, or runtime field changes it.

Loaded reports reject unsupported future versions, the wrong report kind, unknown or malformed suite identity, duplicate or reordered prompt IDs, missing generation configuration, malformed per-case results, unsupported rubric versions, invalid contamination states, absolute identity paths, and digest mismatches.

## Contamination classifications

| JSON status | Meaning | Human held-out status |
| --- | --- | --- |
| `verified_clean` | Exact suite audit coverage and every linked identity check pass | Verified clean |
| `contaminated` | Known historical contamination, a matching finding, or an enabled contamination override | Contaminated |
| `audit_unavailable` | No compatible data-quality report was supplied | Unverified |
| `audit_does_not_cover_suite` | The report does not list the exact suite version and ordered prompt IDs | Unverified |
| `provenance_mismatch` | Report, manifest, and checkpoint data identities do not all match | Unverified |
| `legacy_unverified` | Legacy data provenance cannot establish document-aware audit coverage | Unverified |

A candidate result may use held-out wording only when all of the following are true:

1. The data-quality report is valid and records registry version 1 coverage for the exact candidate suite version and ordered prompt IDs.
2. The report contains no contamination finding for that suite.
3. The contamination override is false.
4. A valid document-aware manifest-v2 links the exact data-quality report digest.
5. The checkpoint's recorded data-manifest digest equals that manifest digest.

If any condition is absent, the result remains candidate/unverified or contaminated. A generic zero contamination count, a report for another suite, or a paraphrased prompt is insufficient.

## Generation benchmark reports

Generation benchmarks are separate from evaluation scores. The benchmark script preserves `perf_counter` timing, synchronizes CUDA around warm-up and measured regions, excludes warm-ups from aggregates, and records generated tokens for every measured run.

```powershell
python scripts/benchmark_generation.py `
  --checkpoint checkpoints\anchor_v2_3_finetuned.pt `
  --attention-backend manual `
  --seed 1337 `
  --warmup-runs 2 `
  --runs 10 `
  --output-json runs\benchmarks\generation-v2.json
```

Benchmark report version 2 records the resolved `manual` or `sdpa` attention backend in the canonical configuration identity in addition to the version-1 fields: benchmark version; seed; path-safe checkpoint, model, and tokenizer identities; device, dtype, and actual compile status; warm-up and measured-run counts; prompt digest and input-token count; generation settings; ordered per-run elapsed time, generated tokens, and throughput; aggregate latency and tokens per second; valid peak CUDA memory when applicable; warnings; and a canonical digest. The report loader continues to validate version-1 reports and interprets their absent backend as `manual`.

Manual attention is the default. `--attention-backend sdpa` requests PyTorch SDPA and fails if the API is unavailable; `auto` uses SDPA when available and otherwise resolves to manual. Availability and performance depend on the PyTorch build, device, dtype, shape, and internal kernel selection. Reports identify the resolved backend without claiming a particular kernel or universal speedup. Checkpoint weights remain compatible between backends, while exact training resume requires the same backend.

CPU reports use `null` for peak CUDA memory. CUDA reports require a valid measured value. Warm-up results never enter measured aggregates. Timing values and report digests that contain them are environment-dependent; they are not reproducible performance claims across unlike machines. ByteSeed does not compare these measurements with external models or claim benchmark superiority.

## Current claims

Infrastructure for deterministic candidate evaluation now exists, but no verified candidate-suite score is published here. Until an actual run has exact clean audit and checkpoint provenance, the honest project-level statement remains:

- Anchor-retention regression: 9/9.
- Held-out generalization: not yet measured.
