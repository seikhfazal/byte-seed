from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .checkpoint import (
    CheckpointCompatibilityError,
    CheckpointKind,
    CheckpointOperation,
    LoadedCheckpoint,
    build_checkpoint,
    build_resume_state,
    move_optimizer_state_to_device,
    restore_rng_state,
    restore_scaler_state,
    select_checkpoint,
    training_config_snapshot,
    validate_exact_resume_checkpoint,
    validate_training_config,
)
from .config import align_config_to_tokenizer, config_from_checkpoint, load_config
from .dataset import load_processed
from .model import GPT
from .provenance import build_checkpoint_provenance, build_pretraining_data_manifest
from .tokenizer import ByteSeedTokenizer
from .utils import ensure_dir, set_seed


@torch.no_grad()
def estimate_loss(model: GPT, train_data, val_data, cfg) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = data.get_batch(cfg.batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def learning_rate(step: int, cfg) -> float:
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_iters)
    return cfg.learning_rate


def next_iteration(completed_iteration: int) -> int:
    """Return the next step after a checkpointed, fully completed optimizer step."""
    if isinstance(completed_iteration, bool) or not isinstance(completed_iteration, int):
        raise ValueError("completed_iteration must be an integer.")
    if completed_iteration < 0:
        raise ValueError("completed_iteration must be non-negative.")
    return completed_iteration + 1


def should_evaluate(step: int, cfg) -> bool:
    return step % cfg.eval_interval == 0 or step == cfg.max_iters - 1


def update_early_stopping(
    validation_loss: float,
    best_val: float,
    patience_left: int,
    configured_patience: int,
) -> tuple[float, int, bool, bool]:
    """Apply the existing strict-improvement rule and return updated control state."""
    if validation_loss < best_val:
        return validation_loss, configured_patience, True, False
    if configured_patience > 0:
        patience_left = max(0, patience_left - 1)
        return best_val, patience_left, False, patience_left == 0
    return best_val, patience_left, False, False


def resolve_resume_checkpoint(
    checkpoint_dir: str | Path,
    *,
    explicit_path: str | Path | None,
    allow_inexact_resume: bool,
    runtime_tokenizer_identity: dict[str, Any],
    runtime_data_manifest: dict[str, Any],
) -> tuple[LoadedCheckpoint | None, dict[str, Any] | None]:
    """Resolve provenance-verified exact resume; allow explicit inexact continuation."""
    if allow_inexact_resume and explicit_path is None:
        raise ValueError(
            "--allow-inexact-resume requires --resume-checkpoint with an explicit path."
        )

    if explicit_path is not None:
        # Structural loading validates any known tokenizer identity first. A known
        # tokenizer mismatch is never eligible for the inexact-resume exception.
        selected = select_checkpoint(
            checkpoint_dir,
            CheckpointOperation.PRETRAIN_RESUME,
            explicit_path=explicit_path,
            runtime_tokenizer_identity=runtime_tokenizer_identity,
        )
        assert selected is not None
        try:
            resume_state = dict(
                validate_exact_resume_checkpoint(
                    selected.data,
                    runtime_tokenizer_identity=runtime_tokenizer_identity,
                    runtime_data_manifest=runtime_data_manifest,
                )
            )
        except CheckpointCompatibilityError as exc:
            if not allow_inexact_resume:
                raise CheckpointCompatibilityError(
                    f"Checkpoint {selected.path} is only an inexact pretraining continuation: "
                    f"{exc} Use --allow-inexact-resume with this explicit path to accept "
                    "missing execution/data provenance or a changed data manifest."
                ) from exc
            print(
                "WARNING: inexact pretraining resume explicitly enabled. "
                f"The continuation is not exact because: {exc}"
            )
            return selected, None
        return selected, resume_state

    selected = select_checkpoint(
        checkpoint_dir,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
        runtime_tokenizer_identity=runtime_tokenizer_identity,
        runtime_data_manifest=runtime_data_manifest,
    )
    if selected is not None:
        return selected, dict(
            validate_exact_resume_checkpoint(
                selected.data,
                runtime_tokenizer_identity=runtime_tokenizer_identity,
                runtime_data_manifest=runtime_data_manifest,
            )
        )

    partial = select_checkpoint(
        checkpoint_dir,
        CheckpointOperation.PRETRAIN_RESUME,
        runtime_tokenizer_identity=runtime_tokenizer_identity,
    )
    if partial is not None:
        raise CheckpointCompatibilityError(
            "Automatic resume found structurally resumable checkpoints but none with complete, "
            "matching execution and data provenance. Automatic resume never downgrades to "
            f"inexact continuation; choose an explicit path such as {partial.path} together "
            "with --allow-inexact-resume if that tradeoff is intentional."
        )

    # If structural candidates exist only without runtime tokenizer validation, they
    # are known tokenizer mismatches (or malformed provenance), not a clean no-resume case.
    incompatible = select_checkpoint(
        checkpoint_dir,
        CheckpointOperation.PRETRAIN_RESUME,
    )
    if incompatible is not None:
        raise CheckpointCompatibilityError(
            "Automatic resume found pretraining checkpoints, but none is compatible with "
            "the current tokenizer identity. Automatic resume did not start fresh or "
            "silently downgrade."
        )
    return None, None


def train(
    config_path: str,
    resume: bool = False,
    max_iters: int | None = None,
    resume_checkpoint: str | None = None,
    allow_inexact_resume: bool = False,
) -> Path:
    cfg = load_config(config_path, {"max_iters": max_iters})
    if allow_inexact_resume and resume_checkpoint is None:
        raise ValueError(
            "--allow-inexact-resume requires --resume-checkpoint with an explicit path."
        )
    set_seed(cfg.seed)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    cfg = align_config_to_tokenizer(cfg, tokenizer)
    requested_cfg = cfg

    # Compute immutable provenance once. Selection, validation, and every checkpoint
    # save reuse these records; large token arrays are never rehashed in the loop.
    runtime_tokenizer_identity = tokenizer.identity
    runtime_data_manifest = build_pretraining_data_manifest(
        cfg.processed_data_dir,
        tokenizer_identity=runtime_tokenizer_identity,
        train_split=cfg.train_split,
    )
    checkpoint_provenance = build_checkpoint_provenance(
        runtime_tokenizer_identity,
        data_manifest=runtime_data_manifest,
    )

    checkpoint_dir = ensure_dir(cfg.checkpoint_dir)
    ckpt = None
    exact_resume_state = None
    start_iter = 0
    best_val = float("inf")

    if resume or resume_checkpoint is not None:
        selected, exact_resume_state = resolve_resume_checkpoint(
            checkpoint_dir,
            explicit_path=resume_checkpoint,
            allow_inexact_resume=allow_inexact_resume,
            runtime_tokenizer_identity=runtime_tokenizer_identity,
            runtime_data_manifest=runtime_data_manifest,
        )
        if selected is not None:
            ckpt_path = selected.path
            ckpt = selected.data
            cfg = config_from_checkpoint(
                ckpt.get("config", cfg.__dict__), fallback_device=cfg.device
            )
            # Runtime artifact locations come from the current invocation. Their bytes and
            # split identity, rather than machine-specific paths, govern exact compatibility.
            cfg.tokenizer_dir = requested_cfg.tokenizer_dir
            cfg.processed_data_dir = requested_cfg.processed_data_dir
            cfg.train_split = requested_cfg.train_split
            if max_iters is not None:
                cfg.max_iters = int(max_iters)
            # iter is the last optimizer step whose evaluation/control updates are complete.
            start_iter = next_iteration(ckpt["iter"])
            best_val = float(ckpt.get("best_val", best_val))
            if exact_resume_state is not None:
                requested_device = requested_cfg.resolved_device
                requested_critical = training_config_snapshot(
                    requested_cfg.__dict__,
                    device_type=requested_device,
                    amp_enabled=requested_device == "cuda",
                )
                validate_training_config(
                    exact_resume_state["training_config"],
                    requested_critical,
                )
            print(f"Resumed from {ckpt_path}; checkpoint config is being used for model shape.")

    device = cfg.resolved_device
    train_data, val_data = load_processed(cfg.processed_data_dir, cfg.block_size, device)
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=device == "cuda")
    critical_config = training_config_snapshot(
        cfg.__dict__,
        device_type=device,
        amp_enabled=scaler.is_enabled(),
    )

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, device)

    patience_left = cfg.early_stopping_patience
    if exact_resume_state is not None:
        validate_training_config(exact_resume_state["training_config"], critical_config)
        restore_scaler_state(scaler, exact_resume_state["amp_scaler"])
        early_stopping = exact_resume_state["early_stopping"]
        best_val = float(early_stopping["best_val"])
        patience_left = int(early_stopping["patience_left"])
        if cfg.early_stopping_patience > 0 and patience_left == 0:
            print("Early stopping had already triggered at the saved continuation point.")
            return checkpoint_dir / "best.pt"

    steps = tqdm(range(start_iter, cfg.max_iters), desc=f"Training {cfg.model_name}")
    if exact_resume_state is not None:
        # Setup above may consume randomness. Restore last, immediately before the first batch draw.
        restore_rng_state(exact_resume_state["rng_state"])

    for step in steps:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            xb, yb = train_data.get_batch(cfg.batch_size)
            with autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(xb, yb)
                loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if should_evaluate(step, cfg):
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"iter {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            best_val, patience_left, improved, should_stop = update_early_stopping(
                losses["val"],
                best_val,
                patience_left,
                cfg.early_stopping_patience,
            )

            # Save one coherent continuation point after this step's scaler/eval/control updates.
            resume_state = build_resume_state(
                scaler=scaler,
                best_val=best_val,
                patience_left=patience_left,
                training_config=critical_config,
            )
            payload = build_checkpoint(
                CheckpointKind.PRETRAIN,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=cfg.__dict__,
                iteration=step,
                best_val=best_val,
                resume_state=resume_state,
                provenance=checkpoint_provenance,
            )
            torch.save(
                payload,
                checkpoint_dir / f"{cfg.model_name.lower()}_iter_{step}.pt",
            )
            if improved:
                torch.save(payload, checkpoint_dir / "best.pt")
            if should_stop:
                print("Early stopping triggered.")
                break
    return checkpoint_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Explicit pretraining-resume checkpoint. Implies --resume and never falls back.",
    )
    parser.add_argument(
        "--allow-inexact-resume",
        action="store_true",
        help=(
            "Allow partial legacy continuation only with --resume-checkpoint; "
            "RNG/scaler/patience state will not be exact."
        ),
    )
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()
    train(
        args.config,
        args.resume,
        args.max_iters,
        args.resume_checkpoint,
        args.allow_inexact_resume,
    )


if __name__ == "__main__":
    main()
