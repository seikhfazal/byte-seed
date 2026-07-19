from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.byteseed.checkpoint import CheckpointOperation, load_checkpoint
from src.byteseed.config import load_config
from src.byteseed.eval_prompts import (
    ANCHOR_RETENTION_PROMPTS,
    ANCHOR_RETENTION_SUITE,
    registered_evaluation_suites,
    get_evaluation_suite,
)
from src.byteseed.evaluation import (
    GenerationConfig,
    logical_checkpoint_identity,
    render_evaluation_report,
    run_evaluation,
    torch_batch_generator,
    write_evaluation_report,
)
from src.byteseed.generate import load_model, marker_id, stop_token_ids
from src.byteseed.tokenizer import ByteSeedTokenizer
from src.byteseed.provenance import sha256_file

PREFERRED_CHECKPOINTS = (
    "checkpoints/anchor_v2_3_finetuned.pt",
    "checkpoints/anchor_v2_2_finetuned.pt",
)

PROMPTS = [prompt.text for prompt in ANCHOR_RETENTION_PROMPTS]


def default_checkpoint() -> str:
    for checkpoint in PREFERRED_CHECKPOINTS:
        if (ROOT / checkpoint).exists():
            return checkpoint
    return PREFERRED_CHECKPOINTS[-1]


def build_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def clean_output(text: str) -> str:
    for marker in ("<|end|>", "<|user|>", "<|assistant|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def generate_answer(model: torch.nn.Module, tokenizer: ByteSeedTokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None) -> str:
    device = next(model.parameters()).device
    formatted = build_prompt(prompt)
    ids = torch.tensor([tokenizer.encode(formatted, add_bos=True)], dtype=torch.long, device=device)
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        vocab_limit=tokenizer.vocab_size,
        stop_token_ids=stops,
    )
    return clean_output(tokenizer.decode(out[0, ids.shape[1] :].tolist()))


def contains_any(text: str, options: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(option in lower for option in options)


def check_answer(prompt: str, answer: str) -> tuple[bool, str]:
    lower = answer.lower().strip()
    if not lower:
        return False, "answer is empty"
    if prompt == "who are you?":
        return ("byteseed" in lower, "contains ByteSeed" if "byteseed" in lower else "missing ByteSeed")
    if prompt == "what is a stack ?":
        ok = "lifo" in lower and "bfs" not in lower
        return ok, "contains LIFO and not BFS" if ok else "must contain LIFO and not contain BFS"
    if prompt == "What is a queue?":
        ok = "fifo" in lower and not ("lifo" in lower and "stack" in lower)
        return ok, "contains FIFO without stack confusion" if ok else "must contain FIFO without stack confusion"
    if prompt == "What is overfitting?":
        ok = "training" in lower and ("validation" in lower or "unseen" in lower or "new data" in lower)
        return ok, "mentions training and validation/unseen/new data" if ok else "must mention training and validation or unseen/new data"
    if prompt == "What is underfitting?":
        simple = contains_any(lower, ("too simple", "not trained enough", "not learn", "too limited"))
        poor_train = "training" in lower or "train" in lower
        poor_val = "validation" in lower
        poor = contains_any(lower, ("poor", "bad", "badly", "high loss"))
        ok = simple and poor and poor_train and poor_val
        return ok, "mentions too simple/not trained enough and poor train/validation performance" if ok else "must mention too simple/not trained enough and poor train/validation performance"
    if prompt == "Help me plan a 1 hour DSA study session.":
        return ("minute" in lower, "contains minutes" if "minute" in lower else "missing minutes")
    if prompt == "How do I run ByteSeed chat?":
        ok = "python chat.py" in lower and "chat.py.py" not in lower
        return ok, "contains python chat.py and not chat.py.py" if ok else "must contain python chat.py and not chat.py.py"
    if prompt == "My PyTorch says CUDA is false. What should I check?":
        ok = "cuda" in lower and ("pytorch" in lower or "torch" in lower or "nvidia" in lower)
        return ok, "mentions CUDA and PyTorch/Torch/NVIDIA" if ok else "must mention CUDA and PyTorch or NVIDIA"
    if prompt == "Should I upload checkpoints to GitHub?":
        ok = "checkpoint" in lower and ("do not commit" in lower or "avoid committing" in lower)
        return ok, "says do not commit/avoid committing checkpoints" if ok else "must say do not commit or avoid committing checkpoints"
    return False, "no check defined"


def _read_json(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {Path(path).name}.")
    return value


def _model_configuration(model: torch.nn.Module) -> dict[str, object]:
    config = model.config
    fields = (
        "model_name",
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "attention_backend",
    )
    return {field: getattr(config, field) for field in fields}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Deterministic ByteSeed retention and candidate-suite evaluation."
    )
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to the newest stable Anchor checkpoint found.",
    )
    parser.add_argument(
        "--suite",
        default=ANCHOR_RETENTION_SUITE,
        choices=[suite.suite_id for suite in registered_evaluation_suites()],
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--attention-backend",
        choices=("manual", "sdpa", "auto"),
        default=None,
        help="Attention implementation. Default: config value (manual when omitted).",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["stochastic", "greedy"],
        default="stochastic",
    )
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--data-quality-report", default=None)
    parser.add_argument("--data-manifest", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        overrides={
            "device": args.device,
            "attention_backend": args.attention_backend,
        },
    )
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    checkpoint = args.checkpoint or default_checkpoint()
    loaded = load_checkpoint(
        checkpoint,
        CheckpointOperation.MODEL_LOAD,
        runtime_tokenizer_identity=tokenizer.identity,
    )
    model = load_model(cfg, checkpoint, tokenizer=tokenizer)
    actual_device = str(next(model.parameters()).device)
    actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    stops = tuple(
        sorted(
            stop_token_ids(
                tokenizer,
                marker_id(tokenizer, "<|end|>") is not None,
            )
        )
    )
    generation = GenerationConfig(
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=stops,
        stop_at_end=True,
        dtype=actual_dtype,
        device=actual_device,
        compile=False,
        batch_size=args.batch_size,
        prompt_format_version=1,
        deterministic_algorithms=args.deterministic_algorithms,
        sampling_mode=args.sampling_mode,
    )

    provenance = loaded.data.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    data_manifest_digest = provenance.get("data_manifest_digest")
    data_manifest_digest = (
        data_manifest_digest if isinstance(data_manifest_digest, str) else None
    )
    data_manifest = _read_json(args.data_manifest)
    if data_manifest is None and isinstance(provenance.get("data_manifest"), Mapping):
        data_manifest = dict(provenance["data_manifest"])
    quality_report = _read_json(args.data_quality_report)
    kind = loaded.info.kind.value if loaded.info.kind is not None else loaded.info.kind_label
    checkpoint_identity = logical_checkpoint_identity(
        checkpoint,
        version=loaded.info.version,
        kind=kind,
        legacy=loaded.info.legacy,
        progress=loaded.info.progress,
        data_manifest_digest=data_manifest_digest,
        artifact_sha256=sha256_file(checkpoint),
    )
    report = run_evaluation(
        get_evaluation_suite(args.suite),
        generation,
        torch_batch_generator(model, tokenizer),
        checkpoint_identity=checkpoint_identity,
        model_configuration=_model_configuration(model),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_identity=(tokenizer.identity if loaded.tokenizer_verified else None),
        data_manifest_digest=data_manifest_digest,
        quality_report=quality_report,
        data_manifest=data_manifest,
    )
    print(f"checkpoint: {Path(checkpoint).name}")
    print(render_evaluation_report(report))
    if args.output_json:
        write_evaluation_report(
            args.output_json,
            report,
            overwrite=args.overwrite,
        )
        print(f"Evaluation report written: {Path(args.output_json).name}")


if __name__ == "__main__":
    main()

