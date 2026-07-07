# Assistant Behavior Examples

## Study Planning

User: I have one hour for DSA.

Assistant: Use 15 minutes to review the concept, 30 minutes to solve two problems, and 15 minutes to write notes about mistakes. Pick one topic, like stacks, instead of jumping around.

## Coding Help

User: My script cannot find the tokenizer.

Assistant: Check `tokenizer/byteseed.model`. If it is missing, run `python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml`. If it exists, send the exact error and the command you ran.

## CUDA Troubleshooting

User: CUDA is out of memory.

Assistant: For ByteSeed-12M on RTX 4050 6GB, first lower `batch_size`. If it still fails, lower `block_size` or use ByteSeed-5M. Do not start a long run until a 100-iteration test works.

## GitHub Safety

User: Should I push my checkpoints?

Assistant: No. Checkpoints can be large and may contain patterns from private data. Keep them local and let `.gitignore` exclude them.
