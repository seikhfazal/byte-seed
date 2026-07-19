# Testing ByteSeed

ByteSeed's automated suite is deterministic, CPU-only, and intentionally uses tiny in-memory models plus temporary files. It does not require a local checkpoint, the ignored SentencePiece tokenizer binaries, a generated dataset, CUDA, or internet access while tests run.

## Local setup

Create or activate a Python 3.11 environment, then install the project and test dependency:

```powershell
python -m pip install -e . -r requirements-dev.txt
```

Run the complete suite:

```powershell
python -m compileall -q src scripts chat.py tests
python -m pytest -q
```

Tests reset Python, NumPy, and Torch CPU RNG state before every test. CUDA RNG state is also seeded when CUDA is available, but CI remains CPU-only. Tests must not depend on execution order or assert a sampled natural-language response.

## Known strict expected failures

The suite records confirmed v0.4 audit defects with focused `strict=True` expected failures. An unexpected pass is therefore a CI failure until the corresponding defect-fix PR changes the test into a normal passing regression test.

PR 2 fixed the two audit CD-01 failures: SFT truncation now preserves supervised assistant targets, and all-ignored model targets are rejected with a clear error.

PR 2B fixed the audit CD-02 TokenDataset minimum-length sampling bound; its strict xfail is now a passing regression.

PR 2C fixed audit CD-08: batched generation now tracks stop-token completion independently for each row.

PR 3 adds deterministic coverage for checkpoint schema metadata, kind-aware pretraining resume selection, explicit-path validation, corrupt-candidate handling, stable ordering, and legacy inference compatibility. SFT and model-only checkpoints cannot be selected for pretraining resume.

PR 4 adds deterministic CPU coverage for exact pretraining continuation state. Tests capture and restore Python and PyTorch CPU RNG state, exercise optional all-device CUDA RNG handling with test doubles, round-trip enabled and disabled scaler state, preserve early-stopping patience, validate training-critical configuration, and migrate nested optimizer tensor state. A tiny dropout/AdamW test compares uninterrupted training with an interrupted, serialized, reconstructed, and resumed run; final parameters, optimizer state, progress, and subsequent random samples must match. Legacy Anchor-like inference fixtures remain covered, while partial legacy pretraining continuation requires explicit opt-in.

PR 5 adds deterministic coverage for bounded, streaming SHA-256 file hashing; path-independent tokenizer identity; same-sized tokenizer files with different bytes or special-token IDs; canonical data-manifest ordering and path normalization; and byte, split, and preprocessing changes. Exact-resume tests require matching tokenizer, training-corpus, validation-corpus, and split identity. Automatic selection skips newer mismatched checkpoints, explicit mismatch never falls back, precomputed provenance is reused without rehashing at checkpoint save, and legacy Anchor-like inference remains loadable with an explicit unverified-compatibility warning.

PR 6 adds deterministic coverage for real document identities, conservative Unicode/newline/whitespace normalization, exact and canonical duplicate grouping, order-independent hash-based train/validation assignment, and post-split leakage rejection. Duplicate groups cannot cross splits. Tests cover exact normalized evaluation-prompt matches in plain text and structured SFT fields, default rejection plus the explicit historical override, deterministic machine-readable quality reports, and manifest-v1/v2 compatibility. A temporary end-to-end preparation fixture proves that documents are split before independent BOS/EOS tokenization and that the quality-report digest participates in exact-resume provenance. The historical Anchor prompts are tested as known SFT overlap, not held-out inputs.

PR 7 adds deterministic CPU coverage for stable suite IDs, unchanged Anchor prompt text/order, candidate paraphrase identity, transparent rubric normalization, required/forbidden concepts, and visible human-review or unscored cases. Tiny generator doubles prove fixed-seed stochastic repeatability, greedy output independence from the sampling seed, ordered batch handling, explicit generation settings, and finally-safe restoration of Python, Torch CPU, and deterministic-algorithm state after success or failure. Report tests cover canonical evaluation and benchmark digests, strict future-version/kind/schema rejection, UTF-8 writing, no silent overwrite, path-safe identities, exact audit coverage, manifest/report/checkpoint linkage, contamination overrides, synchronized benchmark timing, excluded warm-ups, aggregate throughput, and CPU/CUDA memory-field rules. CUDA synchronization is exercised through monkeypatching; no CUDA runtime, real checkpoint, generated dataset, or real performance measurement is required.

PR 8 adds deterministic CPU coverage for the default manual backend, explicit `manual`/`sdpa`/`auto` selection, capability-based fallback, and clear unavailable/invalid failures. FP32 tests compare manual and SDPA attention outputs, full-model logits and losses, input gradients, QKV/output-projection gradients, causal-prefix behavior, zero-dropout determinism, and controlled generation. Tests also prove identical parameter counts and state-dict keys, bidirectional checkpoint-weight loading, training-only SDPA dropout, exact-resume backend validation, legacy-manual defaults, CLI parsing, benchmark report v1 compatibility, and backend-aware v2 report identity. They do not assert identical random dropout masks or require CUDA.

PR 9 adds CPU-only coverage for inference cache structure and validation, absolute-position offsets, manual and SDPA prefill/incremental logits, multi-step cache growth, batch generation, stop tokens, sampling settings, and controlled greedy equivalence at strict FP32 tolerances. Rollover fixtures prove that cache reuse stops when learned-position context cropping begins and that output remains equal to the uncached reference path. Tests also prove request isolation, unchanged parameters/state-dict/checkpoint schema, cache-off CLI defaults, compile/cache rejection, benchmark report v3 identity, and continued version-1/version-2 report validation as uncached. No timing result or speedup is asserted.

The suite currently has no known strict expected failures.

Later defect-fix PRs must remove the relevant `xfail` marker, retain the focused assertion, and add any boundary coverage needed for the corrected behavior. Do not silently turn a known defect into a passing compatibility expectation.

## CI scope

GitHub Actions runs on `ubuntu-latest` with Python 3.11 and executes:

```text
python -m compileall -q src scripts chat.py
python -m pytest -q
```

CI deliberately does not train or pretrain; run SFT; build or import data; train a tokenizer; use CUDA; benchmark GPUs; create persistent checkpoints; load local repository checkpoints; or require ignored tokenizer binaries. Checkpoint round-trip tests use only temporary CPU fixtures. Those workflows need separate, explicitly authorized validation.
