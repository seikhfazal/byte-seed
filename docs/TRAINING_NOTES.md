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
- Anchor datasets worked better than broad curated SFT for the current model size.

Useful anchor checkpoints produced during development:

- `checkpoints/anchor_finetuned.pt`
- `checkpoints/anchor_v2_finetuned.pt`
- `checkpoints/anchor_v2_1_finetuned.pt`
- `checkpoints/anchor_v2_2_finetuned.pt`

## Current Stable Checkpoint

`checkpoints/anchor_v2_2_finetuned.pt` is the current stable checkpoint.

Anchor v2.2 focused on cleaning label artifacts, improving the demo prompts, and keeping answers short and direct.

## Chat Mode

Stateless chat became the default because the model was trained mostly on single-turn examples. Multi-turn history can confuse the tiny model, so history is off unless explicitly enabled with `/history on`.

Broad curated SFT is not recommended right now for this checkpoint family.

## Anchor v2.3 Local Patch

Anchor v2.3 is a tiny targeted local patch for underfitting wording and CUDA false troubleshooting confusion. It starts from checkpoints/anchor_v2_2_finetuned.pt and writes checkpoints/anchor_v2_3_finetuned.pt. It is not a broad model improvement, and the same small-model limitations still apply.

