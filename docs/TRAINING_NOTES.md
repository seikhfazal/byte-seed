# Training Notes

ByteSeed started from a tiny GPT-style decoder-only architecture implemented directly in PyTorch.

## Early Fixes

- The initial tokenizer/config setup had a vocab mismatch: the config used `vocab_size: 8000`, while the actual tiny tokenizer vocabulary was much smaller.
- Effective vocab alignment was added so model loading and generation use the tokenizer/checkpoint shape correctly.
- A YAML numeric coercion issue was fixed where `learning_rate: 3e-4` could load as a string instead of a float.
- SentencePiece special tokens were added:
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
  - `<|end|>`

## SFT Lessons

- Initial broad curated SFT degraded behavior on this tiny model.
- Example-wise SFT fixed the old random chunk SFT problem where prompt/answer boundaries could be mixed across examples.
- SFT truncation now removes overlong prompt context from the left before assistant supervision is discarded, and truncates overlong answers from the right while retaining supervised answer tokens.
- Every accepted SFT example retains at least one supervised assistant target; batches with only ignored (`-100`) targets are rejected with a clear error.
- Existing checkpoints remain compatible because this validation does not change model parameters, architecture, or state-dict structure.
- TokenDataset now supports the minimum valid token count of `block_size + 1`; its sampling bound includes every valid start index while preserving the same shifted integer tensor semantics.
- Inference now tracks completion independently for each batch row. Completed rows append their own stop token as inert filler while unfinished rows continue, without changing model architecture or checkpoint compatibility.
- Anchor datasets worked better than broad curated SFT for the current model size.

Useful anchor checkpoints produced during development:

- `checkpoints/anchor_finetuned.pt`
- `checkpoints/anchor_v2_finetuned.pt`
- `checkpoints/anchor_v2_1_finetuned.pt`
- `checkpoints/anchor_v2_2_finetuned.pt`
- `checkpoints/anchor_v2_3_finetuned.pt`

## Checkpoint Metadata, Provenance, Selection, And Exact Pretraining Resume

- Checkpoint container schema version `1` identifies each checkpoint as `pretrain`, `sft`, or `model_only`.
- Checkpoint provenance version `1` uses SHA-256 identities. The checkpoint container remains schema version `1`; model state-dict keys and architecture are unchanged.
- `tokenizer/byteseed.model` is authoritative because its bytes define the SentencePiece token-to-ID mapping. Tokenizer identity version `1` records its byte size and SHA-256, the effective vocabulary size, all required special-token IDs, and a canonical digest. The optional `.vocab` text file is not required for inference compatibility.
- Historical data-manifest version `1` fingerprints `train.npy`, `val.npy`, tokenizer identity, BOS/EOS handling, and the original contiguous token-fraction split. Document-aware builds use manifest version `2`, which preserves those artifact fingerprints and additionally identifies document format, normalization, deduplication, hash-based split seed/ratio, contamination policy, and the deterministic quality-report digest. Version `1` remains valid only under its original semantics and is never reinterpreted as document-aware.
- The combined manifest digest is SHA-256 over compact UTF-8 JSON with sorted keys, deterministic artifact ordering, normalized forward-slash logical names, and `NaN` disabled. Absolute paths, filesystem timestamps, and informational metadata do not participate in identity.
- New production pretraining checkpoints contain a nested exact-resume block with resume-state version `1`. SFT and model-only checkpoints are not marked exact-pretraining-resumable.
- Automatic pretraining resume selects only checkpoints with complete execution state and matching tokenizer/data provenance. A newer SFT, model-only, partial legacy, tokenizer-mismatched, or corpus-mismatched checkpoint cannot displace a compatible exact pretraining checkpoint, and automatic selection never silently downgrades to inexact continuation.
- The exact state contains Python RNG state, PyTorch CPU RNG state, all initialized CUDA-device RNG states when CUDA is active, AMP GradScaler enablement/state, best validation loss, remaining early-stopping patience, and a training-critical configuration snapshot.
- Pretraining currently has no dedicated `torch.Generator`, and NumPy is used to load processed arrays rather than to make random training decisions, so no separate generator or NumPy RNG state is serialized.
- `iter` means the last fully completed optimizer step, including any evaluation and early-stopping update associated with that step. Resume starts at `iter + 1`, preserving learning-rate, evaluation, and checkpoint cadence without repeating or skipping a step.
- A checkpoint is captured after `optimizer.step()`, `GradScaler.update()`, evaluation, and early-stopping state updates. Its RNG state therefore describes the next operation after that coherent continuation point.
- On exact resume, ByteSeed constructs the dataset, model, optimizer, and scaler; loads model/optimizer/scaler state; moves nested optimizer tensors to the parameter device; validates configuration; and restores RNG state last, immediately before the first resumed batch draw.
- Exact resume validates architecture and training settings, including block/model dimensions, dropout, attention backend, batch and accumulation sizes, AdamW settings, learning rate/schedule, weight decay, warm-up, evaluation cadence, maximum iterations, seed, device type, AMP mode, and early-stopping patience. Machine-specific data paths are not identity fields; tokenizer bytes, corpus bytes, and split/preprocessing identity are validated through the manifest instead. Differing critical fields fail clearly. Resume metadata created before the backend field is interpreted as `manual`, because that was the only implementation available.
- `--resume-checkpoint` selects one explicit checkpoint and never falls back. A PR 4 state-complete checkpoint without provenance, a partial PR 3 checkpoint, or a legacy pretraining checkpoint fails exact resume by default.
- Inexact continuation is available only with both an explicit path and `--allow-inexact-resume`. Missing execution state or missing/data-mismatched provenance produces a prominent warning and is never described as exact. A known tokenizer mismatch is always rejected, even with this opt-in.
- Legacy Anchor-like checkpoints remain loadable for inference and are identified with a focused warning as unverified rather than cryptographically compatible. Structurally complete legacy pretraining checkpoints remain recognizable for explicit inexact continuation.
- New pretraining checkpoints store tokenizer identity, the complete data manifest, and its combined digest. New SFT checkpoints store tokenizer identity. `model_only` checkpoints can store tokenizer identity when their save caller has it; no production model-only save path fabricates provenance.
- Tokenizer and data identities are computed once during startup and reused for selection, diagnostics, and all checkpoint saves. They are not recomputed per batch, evaluation, checkpoint save, or generated token.
- SFT initializes from compatible model weights; interrupted-SFT exact resume is outside this pretraining-only change.

Exact automatic resume:

```powershell
python -m byteseed.pretrain --config configs/byteseed_12m.yaml --resume
```

Explicit inexact continuation from a partial legacy checkpoint:

```powershell
python -m byteseed.pretrain --config configs/byteseed_12m.yaml --resume-checkpoint checkpoints\legacy_pretrain.pt --allow-inexact-resume
```

The exact-resume guarantee is intentionally bounded: with the same supported software/hardware conditions, deterministic operations, matching training-critical configuration, and matching tokenizer/data manifests, continuation restores the same next stochastic and optimizer state. It does not promise bitwise identity across CPU and CUDA, different GPUs, different PyTorch/CUDA versions, nondeterministic kernels, or changed data/tokenizers.

### Attention backend

Manual attention is the backward-compatible default and educational reference. `--attention-backend sdpa` selects `torch.nn.functional.scaled_dot_product_attention` and fails clearly if the installed PyTorch build does not expose it. `--attention-backend auto` selects SDPA when the API is available and otherwise falls back to manual; startup diagnostics and reports record the resolved backend. PyTorch controls the internal kernel choice, which can vary with device, dtype, build, and tensor shape, so SDPA is not claimed to be universally faster or to imply a particular CUDA kernel.

The two paths use the same QKV/output projections, tensor shapes, parameters, state-dict keys, and checkpoint weights. Manual and SDPA checkpoints are therefore mutually loadable for inference or an explicitly chosen new run. Exact pretraining continuation is different: operation ordering and dropout RNG consumption are backend-sensitive, so exact resume requires the same resolved backend. An explicit `--allow-inexact-resume` with an explicit checkpoint may change only this execution choice when all other critical settings match, and emits a warning that continuation is not exact.

The backend option is available on pretraining, SFT, generation, chat, stable evaluation, and generation-benchmark entry points. Evaluation and generation behavior outside attention execution is unchanged.

### Inference-only KV cache

Generation and chat accept --kv-cache as an explicit execution option; it is off by default and is not part of ByteSeedConfig, checkpoint compatibility, training, SFT, or exact-resume state. Manual and SDPA attention both use the same request-local tuple of one key/value tensor pair per layer. Prompt prefill records the full active context, and each subsequent decode step projects only the newest token while the cache remains valid.

Learned absolute position indices begin at zero during prefill and advance from the validated cache length. When prompt plus generated tokens fill block_size and the context window would slide, ByteSeed invalidates the cache and remains on the established cropped, uncached path for the rest of that request. It does not evict old cache entries because their surviving representations contain the prior absolute positions. Cache tensors are discarded after each generate call and are never stored in checkpoints or reused across chat turns.

The current inference wrapper rejects --compile together with --kv-cache because growing cache shapes have not been established as safe for its experimental compile path. Actual cached/uncached performance depends on device, dtype, attention backend, sequence length, and workload; no universal speedup is claimed.

## Document-Aware Data Preparation

- New preparation reads real Markdown-file and JSONL-record boundaries before tokenization. It uses top-level Markdown, `personal_assistant/*.md`, and locally present `generated/markdown/*.md` sources while excluding the historical combined corpus.
- Explicit document IDs are preserved. Otherwise IDs are derived deterministically from logical source plus canonical content; absolute paths and input positions are excluded.
- Normalization version `1` applies Unicode NFC, newline normalization, outer-space removal, and conservative prose-whitespace collapse while preserving indented and fenced code.
- Raw duplicates and additional canonical whitespace/Unicode-equivalent duplicates are reported separately. One representative is selected deterministically, and every duplicate group receives one split assignment.
- Split-strategy version `1` hashes the duplicate-group fingerprint with the configured seed and validation ratio. Assignment is independent of input and filesystem order. Cross-split document-ID, raw-fingerprint, and canonical-fingerprint reuse is rejected.
- Train and validation documents are tokenized independently with the existing BOS/EOS tokens, so one document cannot contribute tokens to both arrays.
- Registered evaluation-prompt contamination is an error by default. `--allow-eval-contamination` exists only for explicit historical reproduction, emits a warning, and changes report/manifest identity. Known contaminated results are never marked held out.
- `data_quality_report.json` version `1` records policy versions, duplicate and split counts, removed IDs/sources, contamination findings, exact registered suite/prompt audit coverage, leakage status, token counts, override status, and a canonical SHA-256 digest without document bodies, timestamps, or absolute paths. Older version-1 reports without the additive coverage field remain valid but cannot prove a candidate suite clean.
- Existing SFT JSONL can be inspected with the non-writing `audit_sft_file` helper for malformed records, empty required fields, duplicate conversations, and registered-prompt overlap. SFT training objectives and PR 2 truncation behavior are unchanged.
- Old contiguous arrays and version-1 manifests remain historical/legacy data. See [DATA_QUALITY.md](DATA_QUALITY.md) for the full contract.

Historical evaluation must be reported exactly as:

- Anchor-retention regression: 9/9.
- Held-out generalization: not yet measured.

All nine stable prompts occur verbatim in Anchor v2.3 SFT material. PR 6 guards new preparation but does not rewrite or retroactively cleanse historical artifacts.

## Evaluation Reports And Determinism

Evaluation does not participate in training or exact-resume state. The shared evaluation runner loads an existing model, records all decoding settings, generates in stable suite order, restores the caller's Python and Torch RNG state in a finally-safe path, and optionally writes an evaluation report. It does not update weights, optimizers, schedulers, early-stopping state, checkpoints, tokenizer files, or datasets.

The historical `anchor-retention-v0.2` suite remains retention-only and known contaminated. The `candidate-paraphrase-v1` suite is initially candidate/unverified. It may be described as held out only for a run whose valid quality report records the exact suite version and ordered prompt IDs, has no matching finding or contamination override, links through the exact document-aware manifest-v2 report digest, and matches the checkpoint's recorded data-manifest digest. A zero generic contamination count is insufficient.

Evaluation reports record seed, temperature, `top_k`, maximum generated tokens, repetition penalty, stop-token policy, dtype, device, compile status, batch size, prompt-format version, and deterministic-algorithm status. They also include path-safe checkpoint metadata, verified tokenizer identity, data-manifest identity, contamination classification, ordered per-case outputs, transparent rubric results, aggregates, and a canonical SHA-256 digest when those identities are available.

Fixed-seed stochastic output is reproducible only within the same supported software, hardware, device, dtype, resolved attention backend, and deterministic-kernel boundary. No bitwise guarantee is made across CPU/CUDA, GPU models, PyTorch/CUDA versions, attention backends, or nondeterministic kernels. Generation benchmark reports record the same configuration boundary but contain environment-dependent timing measurements.

Use `python scripts/eval_stable_v0_2.py --help` for evaluation/report options and `python scripts/benchmark_generation.py --help` for benchmark-report options. See [EVALUATION.md](EVALUATION.md) for suite semantics, contamination classifications, report schemas, and complete commands.

## Generalization SFT v1 data-only preparation

generalization-sft-v1 adds 768 deterministic examples across 12 balanced
concept families. It is intended to be combined with, not replace, the tracked
400-record curated personal-assistant core in a future explicitly authorized
SFT run. This PR does not train a model.

The builder writes current-format SFT JSONL plus linked version-1 SFT manifest
and quality report files under the ignored generated-data directory by default.
Stable groups represent authored lesson/template clusters, not entire concept
families. The 93 groups keep related prompt forms together while the canonical
document-aware split assigns 648 records to training and 120 to validation,
places all 12 families on both sides, and has zero group leakage. The quality
report classifies every internal near finding; sharing a group does not by
itself excuse prohibited wording. The audit also covers exact and near wording
against anchor-retention-v0.2,
candidate-paraphrase-v1, and generalization-holdout-v1.

The new 24-case holdout is excluded from training and begins unverified.
Registration and a clean data build do not measure quality. A future checkpoint
must link exact tokenizer and combined-data provenance before held-out wording
is permitted. See [Generalization SFT v1](GENERALIZATION_SFT_V1.md).

## Current Stable Checkpoint

`checkpoints/anchor_v2_3_finetuned.pt` is the current stable checkpoint.

Anchor v2.2 focused on cleaning label artifacts, improving the demo prompts, and keeping answers short and direct. Anchor v2.3 is a tiny targeted patch for underfitting wording and CUDA false troubleshooting confusion.

## Chat Mode

Stateless chat became the default because the model was trained mostly on single-turn examples. Multi-turn history can confuse the tiny model, so history is off unless explicitly enabled with `/history on`.

Broad curated SFT is not recommended right now for this checkpoint family.

## v0.3-speed Local Work

v0.3-speed adds inference dtype options (`auto`, `fp32`, `fp16`) and expanded benchmark reporting. `auto` uses fp16 on CUDA and fp32 on CPU. `torch.compile` is exposed as an optional experimental flag and is off by default. This work does not train, pretrain, or change model architecture.
