from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from .config import load_config
from .generate import load_model, marker_id, stop_token_ids
from .tokenizer import ByteSeedTokenizer
from .utils import latest_checkpoint

COMMANDS = "/reset  /history [on|off]  /quit  /exit  /temp <val>  /topk <val>  /max <val>  /raw  /help"


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def format_params(count: int) -> str:
    return f"{count:,}"


def resolve_checkpoint_label(config_path: str, checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    cfg = load_config(config_path)
    path = latest_checkpoint(cfg.checkpoint_dir)
    return str(path) if path is not None else "latest checkpoint"


def show_help() -> None:
    print("Commands:")
    print("  /reset         clear conversation history")
    print("  /history       show history mode")
    print("  /history on    enable history mode, keeping at most 2 turns")
    print("  /history off   disable history mode and use stateless single-turn prompts")
    print("  /quit, /exit   quit chat")
    print("  /temp <val>    set sampling temperature, for example /temp 0.5")
    print("  /topk <val>    set top_k sampling, for example /topk 20")
    print("  /max <val>     set max new tokens, for example /max 150")
    print("  /raw           toggle raw generated text before cleanup")
    print("  /help          show this help")


def print_banner(
    model_name: str,
    params: int,
    device: torch.device,
    checkpoint: str,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
    history_enabled: bool,
) -> None:
    line = "=" * 60
    top_k_text = "none" if top_k is None else str(top_k)
    print(line)
    print("                 ByteSeed Chat")
    print(line)
    print(f"model: {model_name}")
    print(f"params: {format_params(params)}")
    print(f"device: {device.type}")
    print(f"ckpt: {checkpoint}")
    print(f"temp: {temperature:g} | top_k: {top_k_text} | max_new: {max_new_tokens}")
    print(f"history: {'on' if history_enabled else 'off'}")
    print(f"commands: {COMMANDS}")
    print(line)


def build_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    parts: list[str] = []
    for user, assistant in history:
        parts.append(f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>")
    parts.append(f"<|user|>\n{user_message}\n<|assistant|>\n")
    return "\n".join(parts)


def clean_assistant_output(text: str) -> str:
    for marker in ("<|end|>", "<|user|>", "<|assistant|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def parse_positive_int(value: str, name: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid {name}: {value!r}")
        return None
    if parsed <= 0:
        print(f"Invalid {name}: must be greater than 0")
        return None
    return parsed


def parse_float(value: str, name: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        print(f"Invalid {name}: {value!r}")
        return None
    if parsed <= 0:
        print(f"Invalid {name}: must be greater than 0")
        return None
    return parsed


def show_history_mode(settings: dict[str, float | int | bool | None]) -> None:
    print(f"history mode: {'on' if settings.get('history_enabled', False) else 'off'}")


def handle_command(command: str, history: list[tuple[str, str]], settings: dict[str, float | int | bool | None]) -> bool:
    parts = command.split()
    name = parts[0].lower()
    if name in {"/quit", "/exit"}:
        return False
    if name == "/help":
        show_help()
        return True
    if name == "/reset":
        history.clear()
        print("Conversation history cleared.")
        return True
    if name == "/history":
        if len(parts) == 1:
            show_history_mode(settings)
            return True
        if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
            enabled = parts[1].lower() == "on"
            settings["history_enabled"] = enabled
            history.clear()
            print(f"history mode: {'on' if enabled else 'off'}")
            return True
        print("Usage: /history, /history on, or /history off")
        return True
    if name == "/raw":
        settings["raw"] = not bool(settings.get("raw", False))
        print(f"raw mode: {'on' if settings['raw'] else 'off'}")
        return True
    if name == "/temp":
        if len(parts) != 2:
            print("Usage: /temp 0.5")
            return True
        value = parse_float(parts[1], "temperature")
        if value is not None:
            settings["temperature"] = value
            print(f"temperature set to {value:g}")
        return True
    if name == "/topk":
        if len(parts) != 2:
            print("Usage: /topk 20")
            return True
        value = parse_positive_int(parts[1], "top_k")
        if value is not None:
            settings["top_k"] = value
            print(f"top_k set to {value}")
        return True
    if name == "/max":
        if len(parts) != 2:
            print("Usage: /max 150")
            return True
        value = parse_positive_int(parts[1], "max_new_tokens")
        if value is not None:
            settings["max_new_tokens"] = value
            print(f"max_new_tokens set to {value}")
        return True
    print(f"Unknown command: {name}. Type /help for commands.")
    return True


def run_chat(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    checkpoint_label = resolve_checkpoint_label(args.config, args.checkpoint)
    model = load_model(cfg, args.checkpoint)
    device = next(model.parameters()).device
    stop_at_end = marker_id(tokenizer, "<|end|>") is not None
    stops = stop_token_ids(tokenizer, stop_at_end)
    history_enabled = args.history_turns > 0
    settings: dict[str, float | int | bool | None] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "raw": False,
        "history_enabled": history_enabled,
    }
    history: list[tuple[str, str]] = []
    max_history_turns = min(2, max(1, args.history_turns)) if args.history_turns > 0 else 2

    print_banner(
        model.config.model_name,
        parameter_count(model),
        device,
        checkpoint_label,
        args.temperature,
        args.top_k,
        args.max_new_tokens,
        history_enabled,
    )
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.startswith("/"):
            if not handle_command(user, history, settings):
                break
            continue

        use_history = bool(settings.get("history_enabled", False))
        prompt_history = history[-max_history_turns:] if use_history else []
        prompt = build_prompt(prompt_history, user)
        if args.json:
            prompt += "Return a compact JSON object. This mode is experimental.\n"
        input_ids = tokenizer.encode(prompt, add_bos=True)
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        out = model.generate(
            ids,
            max_new_tokens=int(settings["max_new_tokens"] or 1),
            temperature=float(settings["temperature"] or 1.0),
            top_k=int(settings["top_k"]) if settings["top_k"] is not None else None,
            vocab_limit=tokenizer.vocab_size,
            stop_token_ids=stops,
        )
        new_token_ids = out[0, ids.shape[1] :].tolist()
        text = tokenizer.decode(new_token_ids)
        if settings["raw"]:
            print(f"raw: {text!r}")
        assistant = clean_assistant_output(text)
        if assistant:
            print(f"ByteSeed: {assistant}")
        else:
            print("ByteSeed: [empty reply generated; try /temp 0.5 or /max 160]")
        if assistant and bool(settings.get("history_enabled", False)):
            history.append((user, assistant))
            if len(history) > max_history_turns:
                del history[:-max_history_turns]


def build_parser(default_config: str, default_checkpoint: str | None, default_temperature: float, default_top_k: int | None, default_max_new_tokens: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive ByteSeed terminal chat.")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--checkpoint", default=default_checkpoint)
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--top-k", type=int, default=default_top_k)
    parser.add_argument("--history-turns", type=int, default=0, help="Enable startup history mode and keep up to this many previous turns, capped at 2. Default is stateless.")
    parser.add_argument("--json", action="store_true", help="Experimental and unreliable JSON output mode.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: str = "configs/byteseed_12m.yaml",
    default_checkpoint: str | None = None,
    default_temperature: float = 0.3,
    default_top_k: int | None = 8,
    default_max_new_tokens: int = 120,
) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser(default_config, default_checkpoint, default_temperature, default_top_k, default_max_new_tokens)
    args = parser.parse_args(argv)
    run_chat(args)


if __name__ == "__main__":
    main()
