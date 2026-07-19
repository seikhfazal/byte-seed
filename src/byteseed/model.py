from __future__ import annotations

import math
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ATTENTION_BACKENDS, ByteSeedConfig


LayerKVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
KVCache: TypeAlias = tuple[LayerKVCache, ...]


def sdpa_is_available() -> bool:
    """Return whether this PyTorch build exposes the SDPA functional API."""
    return callable(getattr(F, "scaled_dot_product_attention", None))


def resolve_attention_backend(requested: str) -> str:
    """Resolve a configured attention backend to the implementation in use."""
    backend = str(requested).strip().lower()
    if backend not in ATTENTION_BACKENDS:
        choices = ", ".join(ATTENTION_BACKENDS)
        raise ValueError(
            f"Unknown attention backend {requested!r}; expected one of {choices}."
        )
    if backend == "manual":
        return backend
    if sdpa_is_available():
        return "sdpa"
    if backend == "auto":
        return "manual"
    raise RuntimeError(
        "Attention backend 'sdpa' was requested, but this PyTorch build does not "
        "provide torch.nn.functional.scaled_dot_product_attention."
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout_p = float(config.dropout)
        self.attention_backend = resolve_attention_backend(config.attention_backend)
        # Lower-triangular mask blocks attention to future tokens.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        if self.attention_backend == "sdpa":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attention_dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(
                self.mask[:, :, :time, :time] == 0,
                float("-inf"),
            )
            weights = self.dropout(F.softmax(scores, dim=-1))
            out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.proj(out))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        """Run inference attention and return the request-local key/value cache."""
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        incremental = past_key_value is not None
        if incremental:
            past_key, past_value = past_key_value
            k = torch.cat((past_key, k), dim=2)
            v = torch.cat((past_value, v), dim=2)

        if self.attention_backend == "sdpa":
            # A one-token incremental query is the final position, so every
            # cached/current key is visible. Non-square is_causal=True would
            # instead apply a top-left causal mask and hide valid cached keys.
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=not incremental,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            if not incremental:
                scores = scores.masked_fill(
                    self.mask[:, :, :time, :time] == 0,
                    float("-inf"),
                )
            weights = F.softmax(scores, dim=-1)
            out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.proj(out)), (k, v)


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

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        attention, present_key_value = self.attn.forward_with_cache(
            self.ln1(x),
            past_key_value,
        )
        x = x + attention
        x = x + self.mlp(self.ln2(x))
        return x, present_key_value


class GPT(nn.Module):
    def __init__(self, config: ByteSeedConfig):
        super().__init__()
        config.attention_backend = resolve_attention_backend(config.attention_backend)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @property
    def attention_backend(self) -> str:
        return self.config.attention_backend

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

    def _validate_kv_cache(self, idx: torch.Tensor, past_key_values: KVCache) -> int:
        if not isinstance(past_key_values, tuple):
            raise TypeError("past_key_values must be a tuple with one key/value pair per layer.")
        if len(past_key_values) != self.config.n_layer:
            raise ValueError(
                "past_key_values layer count does not match the model: "
                f"expected {self.config.n_layer}, got {len(past_key_values)}."
            )

        expected_batch = idx.size(0)
        expected_heads = self.config.n_head
        expected_head_size = self.config.n_embd // self.config.n_head
        expected_device = self.token_embedding.weight.device
        expected_dtype = self.token_embedding.weight.dtype
        cache_length: int | None = None
        for layer_index, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise TypeError(
                    f"past_key_values[{layer_index}] must be a (key, value) tuple."
                )
            key, value = layer_cache
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"past_key_values[{layer_index}] key and value must be tensors."
                )
            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise ValueError(
                    f"past_key_values[{layer_index}] key/value shapes must match and "
                    "use (batch, heads, sequence, head_dimension)."
                )
            batch, heads, length, head_size = key.shape
            if batch != expected_batch:
                raise ValueError(
                    f"past_key_values[{layer_index}] batch size {batch} does not "
                    f"match input batch size {expected_batch}."
                )
            if heads != expected_heads or head_size != expected_head_size:
                raise ValueError(
                    f"past_key_values[{layer_index}] attention shape does not match "
                    f"{expected_heads} heads of size {expected_head_size}."
                )
            if length < 1 or length > self.config.block_size:
                raise ValueError(
                    f"past_key_values[{layer_index}] sequence length {length} must "
                    f"be between 1 and block_size {self.config.block_size}."
                )
            if key.device != expected_device or value.device != expected_device:
                raise ValueError(
                    f"past_key_values[{layer_index}] must be on model device "
                    f"{expected_device}."
                )
            if key.dtype != expected_dtype or value.dtype != expected_dtype:
                raise ValueError(
                    f"past_key_values[{layer_index}] must use model dtype "
                    f"{expected_dtype}."
                )
            if cache_length is None:
                cache_length = length
            elif length != cache_length:
                raise ValueError("All past_key_values layers must have the same sequence length.")

        assert cache_length is not None
        return cache_length

    @torch.inference_mode()
    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Run inference prefill or one-token decode with an ephemeral KV cache."""
        if self.training:
            raise RuntimeError("KV caching is inference-only; call model.eval() first.")
        if idx.ndim != 2:
            raise ValueError("Cached model input must have shape (batch, sequence).")

        batch, time = idx.shape
        if time < 1:
            raise ValueError("Cached model input must contain at least one token.")
        cache_length = 0
        if past_key_values is not None:
            cache_length = self._validate_kv_cache(idx, past_key_values)
            if time != 1:
                raise ValueError(
                    "Cached incremental decoding supports exactly one new input token."
                )
        if cache_length + time > self.config.block_size:
            raise ValueError(
                f"Cached sequence length {cache_length + time} exceeds block_size "
                f"{self.config.block_size}; invalidate the cache and use cropped "
                "uncached generation."
            )

        pos = torch.arange(cache_length, cache_length + time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        present_key_values: list[LayerKVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            x, layer_present = block.forward_with_cache(x, layer_past)
            present_key_values.append(layer_present)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, tuple(present_key_values)

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
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        self.eval()
        if not isinstance(use_kv_cache, bool):
            raise TypeError("use_kv_cache must be a boolean.")
        stop_ids = {int(token_id) for token_id in stop_token_ids} if stop_token_ids else set()
        stop_id_tensor = (
            torch.tensor(sorted(stop_ids), dtype=idx.dtype, device=idx.device) if stop_ids else None
        )
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        filler_ids = torch.zeros(idx.size(0), dtype=idx.dtype, device=idx.device)
        past_key_values: KVCache | None = None
        cache_active = use_kv_cache
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            if cache_active:
                if (
                    past_key_values is not None
                    and past_key_values[0][0].size(2) >= self.config.block_size
                ):
                    # Learned absolute positions are reassigned when the context
                    # window slides. Reusing cropped cache entries would retain
                    # their old positions, so use the reference path thereafter.
                    past_key_values = None
                    cache_active = False
                    logits, _ = self(idx_cond)
                elif past_key_values is None:
                    logits, past_key_values = self.forward_with_cache(idx_cond)
                else:
                    logits, past_key_values = self.forward_with_cache(
                        idx_cond[:, -1:],
                        past_key_values,
                    )
            else:
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
            if stop_id_tensor is not None:
                # Keep the tensor rectangular: completed rows receive their own stop token as inert filler.
                next_id = torch.where(finished.unsqueeze(1), filler_ids.unsqueeze(1), next_id)
                newly_finished = (~finished) & torch.isin(next_id.squeeze(1), stop_id_tensor)
                filler_ids = torch.where(newly_finished, next_id.squeeze(1), filler_ids)
                finished = finished | newly_finished
            idx = torch.cat((idx, next_id), dim=1)
            if stop_id_tensor is not None and bool(torch.all(finished)):
                break
        return idx







