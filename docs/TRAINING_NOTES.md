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

## Current Stable Checkpoint

`checkpoints/anchor_v2_3_finetuned.pt` is the current stable checkpoint.

Anchor v2.2 focused on cleaning label artifacts, improving the demo prompts, and keeping answers short and direct. Anchor v2.3 is a tiny targeted patch for underfitting wording and CUDA false troubleshooting confusion.

## Chat Mode

Stateless chat became the default because the model was trained mostly on single-turn examples. Multi-turn history can confuse the tiny model, so history is off unless explicitly enabled with `/history on`.

Broad curated SFT is not recommended right now for this checkpoint family.

## v0.3-speed Local Work

v0.3-speed adds inference dtype options (`auto`, `fp32`, `fp16`) and expanded benchmark reporting. `auto` uses fp16 on CUDA and fp32 on CPU. `torch.compile` is exposed as an optional experimental flag and is off by default. This work does not train, pretrain, or change model architecture.
