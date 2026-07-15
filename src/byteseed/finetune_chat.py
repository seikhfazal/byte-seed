from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import align_config_to_tokenizer, load_config
from .generate import load_model, marker_id, stop_token_ids
from .tokenizer import ByteSeedTokenizer
from .utils import ensure_dir, set_seed

IGNORE_INDEX = -100


def format_chat(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


class ChatSFTDataset:
    def __init__(self, path: str | Path, tokenizer: ByteSeedTokenizer, block_size: int, device: str, mask_prompt: bool = True):
        self.block_size = block_size
        self.device = device
        self.pad_id = tokenizer.eos_id if tokenizer.eos_id >= 0 else 0
        self.examples: list[tuple[list[int], list[int]]] = []

        with Path(path).open("r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                x, y = self._encode_example(
                    tokenizer,
                    str(row["user"]),
                    str(row["assistant"]),
                    mask_prompt,
                    example_index=line_number,
                )
                if not x:
                    raise ValueError(f"Example on line {line_number} produced no training tokens.")
                self.examples.append((x, y))

        if not self.examples:
            raise ValueError(f"No SFT examples found in {path}.")

    def _encode_example(
        self,
        tokenizer: ByteSeedTokenizer,
        user: str,
        assistant: str,
        mask_prompt: bool,
        example_index: int | None = None,
    ) -> tuple[list[int], list[int]]:
        prompt_text = f"<|user|>\n{user}\n<|assistant|>\n"
        answer_text = f"{assistant}\n<|end|>"
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
        answer_ids = tokenizer.encode(answer_text, add_bos=False)
        max_tokens = self.block_size + 1
        prompt_token_count = len(prompt_ids)
        answer_token_count = len(answer_ids)

        if max_tokens < 2 or prompt_token_count == 0 or answer_token_count == 0:
            raise self._no_supervision_error(example_index, prompt_token_count, answer_token_count)

        if prompt_token_count + answer_token_count > max_tokens:
            if prompt_token_count < max_tokens:
                prompt_budget = prompt_token_count
                answer_budget = max_tokens - prompt_budget
            else:
                minimum_prompt_tokens = min(prompt_token_count, min(2, max_tokens - 1))
                answer_budget = min(answer_token_count, max_tokens - minimum_prompt_tokens)
                prompt_budget = min(prompt_token_count, max_tokens - answer_budget)

            # Remove old prompt context first while keeping the prompt/assistant boundary suffix.
            prompt_ids = prompt_ids[-prompt_budget:]
            if answer_token_count > answer_budget:
                # Preserve answer content and, when space permits, the final <|end|> token.
                answer_ids = (
                    answer_ids[:answer_budget]
                    if answer_budget == 1
                    else answer_ids[: answer_budget - 1] + answer_ids[-1:]
                )

        tokens = prompt_ids + answer_ids
        x = tokens[:-1]
        y = tokens[1:]

        if mask_prompt:
            prompt_label_count = max(0, min(len(prompt_ids) - 1, len(y)))
            y[:prompt_label_count] = [IGNORE_INDEX] * prompt_label_count

        if not any(label != IGNORE_INDEX for label in y):
            raise self._no_supervision_error(example_index, prompt_token_count, answer_token_count)

        return x, y

    def _no_supervision_error(
        self,
        example_index: int | None,
        prompt_token_count: int,
        answer_token_count: int,
    ) -> ValueError:
        location = f"Example on line {example_index}" if example_index is not None else "SFT example"
        return ValueError(
            f"{location} contains no supervised assistant target tokens after truncation "
            f"(block_size={self.block_size}, prompt_tokens={prompt_token_count}, "
            f"answer_tokens={answer_token_count})."
        )

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self.examples), (batch_size,))
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for index in indices.tolist():
            x, y = self.examples[index]
            pad_len = self.block_size - len(x)
            if pad_len < 0:
                x = x[: self.block_size]
                y = y[: self.block_size]
                pad_len = 0
            x_padded = x + [self.pad_id] * pad_len
            y_padded = y + [IGNORE_INDEX] * pad_len
            xs.append(torch.tensor(x_padded, dtype=torch.long))
            ys.append(torch.tensor(y_padded, dtype=torch.long))
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)


def finetune(config_path: str, checkpoint: str | None, examples: str, iters: int, output: str | None = None, mask_prompt: bool = True) -> Path:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    if checkpoint is None:
        cfg = align_config_to_tokenizer(cfg, tokenizer)
    model = load_model(cfg, checkpoint)
    cfg = model.config
    device = cfg.resolved_device
    data = ChatSFTDataset(examples, tokenizer, cfg.block_size, device, mask_prompt=mask_prompt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate / 3, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=device == "cuda")
    model.train()
    for _ in tqdm(range(iters), desc="Chat fine-tuning"):
        optimizer.zero_grad(set_to_none=True)
        xb, yb = data.get_batch(cfg.batch_size)
        with autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    out = Path(output) if output else ensure_dir(cfg.checkpoint_dir) / "chat_finetuned.pt"
    ensure_dir(out.parent)
    torch.save({"model": model.state_dict(), "config": model.config.__dict__, "iter": iters}, out)
    print(f"Saved chat fine-tuned checkpoint to {out}")
    return out


def generate_reply(model: torch.nn.Module, tokenizer: ByteSeedTokenizer, prompt: str, max_new_tokens: int = 120, temperature: float = 0.3, top_k: int | None = 8) -> str:
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, vocab_limit=tokenizer.vocab_size, stop_token_ids=stops)
    text = tokenizer.decode(out[0, ids.shape[1] :].tolist())
    for marker in ("<|end|>", "<|user|>", "<|assistant|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--examples", default="examples/chat_examples.jsonl")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", default=None, help="Checkpoint output path. Defaults to checkpoints/chat_finetuned.pt.")
    parser.add_argument("--no-mask-prompt", action="store_true", help="Train on prompt tokens too instead of masking them with -100.")
    args = parser.parse_args()
    finetune(args.config, args.checkpoint, args.examples, args.iters, args.output, mask_prompt=not args.no_mask_prompt)


if __name__ == "__main__":
    main()
