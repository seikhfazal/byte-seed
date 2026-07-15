from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ByteSeedConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # Lower-triangular mask blocks attention to future tokens.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual blocks keep training stable for small models.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size {self.config.block_size}.")
        if targets is not None and not torch.any(targets != -100):
            raise ValueError("Targets contain no supervised target tokens; every target is -100.")
        pos = torch.arange(0, time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        vocab_limit: int | None = None,
        stop_token_ids: set[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        stop_ids = {int(token_id) for token_id in stop_token_ids} if stop_token_ids is not None else None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if vocab_limit is not None and vocab_limit < logits.size(-1):
                logits[:, vocab_limit:] = -float("inf")
            if repetition_penalty > 1.0:
                for batch_index in range(idx.size(0)):
                    seen = torch.unique(idx[batch_index])
                    token_logits = logits[batch_index, seen]
                    logits[batch_index, seen] = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_ids is not None and next_id.numel() == 1 and int(next_id.item()) in stop_ids:
                break
        return idx







