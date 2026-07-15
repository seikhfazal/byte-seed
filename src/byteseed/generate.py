from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Any

import torch

from .checkpoint import CheckpointOperation, select_checkpoint
from .config import ByteSeedConfig, config_from_checkpoint, load_config
from .model import GPT
from .tokenizer import ByteSeedTokenizer


NO_CHECKPOINT_MESSAGE = """No checkpoint found. Train first or pass --checkpoint checkpoints/your_file.pt.

Run a short test first:
python -m src.byteseed.pretrain --config configs/byteseed_12m.yaml --max-iters 100

Then generate:
python -m src.byteseed.generate --config configs/byteseed_12m.yaml --prompt "You are ByteSeed Assistant."
"""


def _checkpoint_tensor_shape(ckpt: dict[str, Any]) -> tuple[int | None, int | None]:
    state = ckpt.get("model", {})
    weight = state.get("token_embedding.weight")
    if weight is None:
        return None, None
    return int(weight.shape[0]), int(weight.shape[1])


def _model_cfg_from_checkpoint(base_cfg: ByteSeedConfig, ckpt: dict[str, Any]) -> ByteSeedConfig:
    checkpoint_vocab, checkpoint_embd = _checkpoint_tensor_shape(ckpt)
    if "config" in ckpt:
        model_cfg = config_from_checkpoint(ckpt["config"], fallback_device=base_cfg.device)
    else:
        print("Warning: checkpoint has no saved config. Inferring vocab_size and n_embd from checkpoint weights, using YAML for the rest.")
        model_cfg = base_cfg

    changed = []
    if checkpoint_vocab is not None and model_cfg.vocab_size != checkpoint_vocab:
        changed.append(f"vocab_size {model_cfg.vocab_size} -> {checkpoint_vocab}")
        model_cfg.vocab_size = checkpoint_vocab
    if checkpoint_embd is not None and model_cfg.n_embd != checkpoint_embd:
        changed.append(f"n_embd {model_cfg.n_embd} -> {checkpoint_embd}")
        model_cfg.n_embd = checkpoint_embd
    if changed:
        print(
            "Warning: checkpoint config did not match checkpoint tensor shapes; "
            f"using checkpoint weights as source of truth ({', '.join(changed)})."
        )
        model_cfg.validate()
    return model_cfg


def _shape_mismatch_message(error: RuntimeError, model_cfg: ByteSeedConfig, ckpt: dict[str, Any]) -> RuntimeError:
    checkpoint_vocab, checkpoint_embd = _checkpoint_tensor_shape(ckpt)
    return RuntimeError(
        "Could not load checkpoint because the model shape does not match.\n"
        f"Checkpoint vocab size: {checkpoint_vocab or 'unknown'}\n"
        f"Current model vocab size: {model_cfg.vocab_size}\n"
        f"Checkpoint embedding size: {checkpoint_embd or 'unknown'}\n"
        f"Current embedding size: {model_cfg.n_embd}\n"
        "Likely cause: tokenizer/config/checkpoint mismatch. Use a checkpoint trained with the same tokenizer, "
        "or retrain tokenizer, prepare data, and train a fresh matching checkpoint.\n"
        f"Original PyTorch error: {error}"
    )


def load_model(
    cfg: ByteSeedConfig,
    checkpoint: str | None,
    *,
    tokenizer: ByteSeedTokenizer | None = None,
) -> GPT:
    selected = select_checkpoint(
        cfg.checkpoint_dir,
        CheckpointOperation.MODEL_LOAD,
        explicit_path=checkpoint,
        runtime_tokenizer_identity=tokenizer.identity if tokenizer is not None else None,
    )
    if selected is None:
        raise FileNotFoundError(NO_CHECKPOINT_MESSAGE)
    if tokenizer is not None and selected.tokenizer_verified is False:
        warnings.warn(
            "Checkpoint has no tokenizer fingerprint; legacy inference compatibility is "
            "unverified rather than cryptographically confirmed.",
            RuntimeWarning,
            stacklevel=2,
        )

    ckpt = selected.data
    model_cfg = _model_cfg_from_checkpoint(cfg, ckpt)
    device = model_cfg.resolved_device
    model = GPT(model_cfg).to(device)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise _shape_mismatch_message(exc, model_cfg, ckpt) from exc
    model.eval()
    return model


def marker_id(tokenizer: ByteSeedTokenizer, marker: str) -> int | None:
    token_id = int(tokenizer.sp.piece_to_id(marker))
    if token_id == tokenizer.sp.unk_id() and marker != tokenizer.sp.id_to_piece(token_id):
        return None
    return token_id


def stop_token_ids(tokenizer: ByteSeedTokenizer, stop_at_end: bool) -> set[int]:
    ids: set[int] = set()
    if stop_at_end:
        for marker in ("<|end|>", "<|user|>"):
            token_id = marker_id(tokenizer, marker)
            if token_id is not None:
                ids.add(token_id)
    return ids


def trim_chat_text(text: str, prompt: str = "") -> str:
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :]
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>", 1)[1]
    for marker in ("<|end|>", "<|user|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def complete(
    config_path: str,
    prompt: str,
    checkpoint: str | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    json_mode: bool = False,
    stop_at_end: bool | None = None,
) -> str:
    cfg = load_config(config_path)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, checkpoint, tokenizer=tokenizer)
    device = next(model.parameters()).device
    if json_mode:
        prompt = prompt + "\nReturn a compact JSON object. This mode is experimental.\n"
    if stop_at_end is None:
        stop_at_end = marker_id(tokenizer, "<|end|>") is not None
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        vocab_limit=tokenizer.vocab_size,
        stop_token_ids=stop_token_ids(tokenizer, stop_at_end),
    )
    text = tokenizer.decode(out[0].tolist())
    text = trim_chat_text(text, prompt) if stop_at_end else text
    if json_mode:
        return json.dumps({"experimental_json_text": text}, ensure_ascii=False)
    return text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Experimental and unreliable JSON output mode.")
    parser.add_argument("--stop-at-end", action=argparse.BooleanOptionalAction, default=None, help="Stop and trim at <|end|> or a new <|user|>. Defaults on when the tokenizer has <|end|>.")
    args = parser.parse_args()
    print(complete(args.config, args.prompt, args.checkpoint, args.max_new_tokens, args.temperature, args.top_k, args.json, args.stop_at_end))


if __name__ == "__main__":
    main()
