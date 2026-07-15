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

The only remaining strict expected failure is:

- Batched generation does not stop on stop-token IDs (audit CD-08).

Later defect-fix PRs must remove the relevant `xfail` marker, retain the focused assertion, and add any boundary coverage needed for the corrected behavior. Do not silently turn a known defect into a passing compatibility expectation.

## CI scope

GitHub Actions runs on `ubuntu-latest` with Python 3.11 and executes:

```text
python -m compileall -q src scripts chat.py
python -m pytest -q
```

CI deliberately does not train or pretrain; run SFT; build or import data; train a tokenizer; use CUDA; benchmark GPUs; create persistent checkpoints; load local repository checkpoints; or require ignored tokenizer binaries. Checkpoint round-trip tests use only temporary CPU fixtures. Those workflows need separate, explicitly authorized validation.
