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
- Data-manifest version `1` fingerprints the artifacts pretraining actually consumes: `train.npy` and `val.npy`. Each record contains a logical role/name, byte size, SHA-256, token count, NumPy dtype, and format. The manifest also contains tokenizer identity plus preprocessing identity for BOS/EOS handling and the contiguous token-fraction split. Data preparation writes `data_manifest.json` beside those arrays, while pretraining recomputes the current in-memory contract once from the consumed bytes before checkpoint selection.
- The combined manifest digest is SHA-256 over compact UTF-8 JSON with sorted keys, deterministic artifact ordering, normalized forward-slash logical names, and `NaN` disabled. Absolute paths, filesystem timestamps, and informational metadata do not participate in identity.
- New production pretraining checkpoints contain a nested exact-resume block with resume-state version `1`. SFT and model-only checkpoints are not marked exact-pretraining-resumable.
- Automatic pretraining resume selects only checkpoints with complete execution state and matching tokenizer/data provenance. A newer SFT, model-only, partial legacy, tokenizer-mismatched, or corpus-mismatched checkpoint cannot displace a compatible exact pretraining checkpoint, and automatic selection never silently downgrades to inexact continuation.
- The exact state contains Python RNG state, PyTorch CPU RNG state, all initialized CUDA-device RNG states when CUDA is active, AMP GradScaler enablement/state, best validation loss, remaining early-stopping patience, and a training-critical configuration snapshot.
- Pretraining currently has no dedicated `torch.Generator`, and NumPy is used to load processed arrays rather than to make random training decisions, so no separate generator or NumPy RNG state is serialized.
- `iter` means the last fully completed optimizer step, including any evaluation and early-stopping update associated with that step. Resume starts at `iter + 1`, preserving learning-rate, evaluation, and checkpoint cadence without repeating or skipping a step.
- A checkpoint is captured after `optimizer.step()`, `GradScaler.update()`, evaluation, and early-stopping state updates. Its RNG state therefore describes the next operation after that coherent continuation point.
- On exact resume, ByteSeed constructs the dataset, model, optimizer, and scaler; loads model/optimizer/scaler state; moves nested optimizer tensors to the parameter device; validates configuration; and restores RNG state last, immediately before the first resumed batch draw.
- Exact resume validates architecture and training settings, including block/model dimensions, dropout, batch and accumulation sizes, AdamW settings, learning rate/schedule, weight decay, warm-up, evaluation cadence, maximum iterations, seed, device type, AMP mode, and early-stopping patience. Machine-specific data paths are not identity fields; tokenizer bytes, corpus bytes, and split/preprocessing identity are validated through the manifest instead. Differing critical fields fail clearly.
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

## Current Stable Checkpoint

`checkpoints/anchor_v2_3_finetuned.pt` is the current stable checkpoint.

Anchor v2.2 focused on cleaning label artifacts, improving the demo prompts, and keeping answers short and direct. Anchor v2.3 is a tiny targeted patch for underfitting wording and CUDA false troubleshooting confusion.

## Chat Mode

Stateless chat became the default because the model was trained mostly on single-turn examples. Multi-turn history can confuse the tiny model, so history is off unless explicitly enabled with `/history on`.

Broad curated SFT is not recommended right now for this checkpoint family.

## v0.3-speed Local Work

v0.3-speed adds inference dtype options (`auto`, `fp32`, `fp16`) and expanded benchmark reporting. `auto` uses fp16 on CUDA and fp32 on CPU. `torch.compile` is exposed as an optional experimental flag and is off by default. This work does not train, pretrain, or change model architecture.
