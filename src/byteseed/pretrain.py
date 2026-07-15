from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .checkpoint import CheckpointKind, CheckpointOperation, build_checkpoint, select_checkpoint
from .config import align_config_to_tokenizer, config_from_checkpoint, load_config
from .dataset import load_processed
from .model import GPT
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


def train(
    config_path: str,
    resume: bool = False,
    max_iters: int | None = None,
    resume_checkpoint: str | None = None,
) -> Path:
    cfg = load_config(config_path, {"max_iters": max_iters})
    set_seed(cfg.seed)
    checkpoint_dir = ensure_dir(cfg.checkpoint_dir)
    ckpt = None
    start_iter = 0
    best_val = float("inf")

    if resume or resume_checkpoint is not None:
        selected = select_checkpoint(
            checkpoint_dir,
            CheckpointOperation.PRETRAIN_RESUME,
            explicit_path=resume_checkpoint,
        )
        if selected is not None:
            ckpt_path = selected.path
            ckpt = selected.data
            cfg = config_from_checkpoint(ckpt.get("config", cfg.__dict__), fallback_device=cfg.device)
            if max_iters is not None:
                cfg.max_iters = int(max_iters)
            start_iter = ckpt.get("iter", 0) + 1
            best_val = ckpt.get("best_val", best_val)
            print(f"Resumed from {ckpt_path}; checkpoint config is being used for model shape.")

    if ckpt is None:
        tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
        cfg = align_config_to_tokenizer(cfg, tokenizer)

    device = cfg.resolved_device
    train_data, val_data = load_processed(cfg.processed_data_dir, cfg.block_size, device)
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=device == "cuda")

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    patience_left = cfg.early_stopping_patience
    for step in tqdm(range(start_iter, cfg.max_iters), desc=f"Training {cfg.model_name}"):
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

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"iter {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            ckpt_path = checkpoint_dir / f"{cfg.model_name.lower()}_iter_{step}.pt"
            torch.save(
                build_checkpoint(
                    CheckpointKind.PRETRAIN,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    config=cfg.__dict__,
                    iteration=step,
                    best_val=best_val,
                ),
                ckpt_path,
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                patience_left = cfg.early_stopping_patience
                torch.save(
                    build_checkpoint(
                        CheckpointKind.PRETRAIN,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        config=cfg.__dict__,
                        iteration=step,
                        best_val=best_val,
                    ),
                    checkpoint_dir / "best.pt",
                )
            elif cfg.early_stopping_patience > 0:
                patience_left -= 1
                if patience_left <= 0:
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
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.resume, args.max_iters, args.resume_checkpoint)


if __name__ == "__main__":
    main()
