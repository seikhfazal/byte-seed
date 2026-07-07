# ByteSeed Architecture

ByteSeed is a decoder-only GPT-style language model. It reads token IDs, adds learned token and position embeddings, applies repeated Transformer blocks, and predicts the next token at every position.

Each Transformer block contains:

- LayerNorm
- Multi-head causal self-attention
- Residual connection
- LayerNorm
- MLP with GELU
- Residual connection

The causal attention mask prevents each token from attending to future tokens. This is what makes next-token prediction possible.

The token embedding matrix is tied to the final language modeling head, which reduces parameters and is common in small language models.

