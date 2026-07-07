# Limitations

- ByteSeed is a tiny model, about 11M parameters.
- It is trained on small local, synthetic, and curated datasets.
- It works best in single-turn mode.
- Multi-turn history is disabled by default because the model was mostly trained on single-turn examples.
- It can still hallucinate.
- It can confuse concepts outside its anchor training data.
- It is not safe for factual, medical, legal, financial, or security-critical use.
- It is not a replacement for large LLMs.
- It is best used as a learning project.
