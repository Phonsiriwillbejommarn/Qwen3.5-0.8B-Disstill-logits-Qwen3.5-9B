"""
train_sft.py — Phase 1: Supervised Fine-Tuning (Off-policy warm-up)
Student (Qwen3.5-0.8B-Base) learns ChatML format + reasoning chains
from teacher-generated responses.

Key: SFT ก่อน 1-2 epoch เพื่อลด distribution gap ก่อน distillation
ห้าม overfit — loss บน training set ควรลงมาแต่ไม่ต่ำเกิน 0.5

Usage:
    python train_sft.py
    python train_sft.py --dry_run --max_steps 3
"""

import argparse
import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from functools import partial
from tqdm import tqdm

import config
from utils.model_utils import load_student
from utils.data_utils import build_sft_dataloader


def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine scheduler with linear warmup"""
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps])


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    grad_accum: int,
    max_grad_norm: float,
    epoch: int,
    max_steps: int | None = None,
) -> dict:
    """Train one epoch, return metrics"""
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    step = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} SFT", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum

        loss.backward()

        # Count non-masked tokens
        n_tokens = (labels != -100).sum().item()
        total_loss   += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % config.SFT_LOGGING_STEPS == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

            if max_steps and step >= max_steps:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "steps": step}


def main(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(config.SFT_CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(config.SEED)

    # ── Check data exists ──────────────────────────────────────────────────────
    if not os.path.exists(config.SFT_DATA_PATH):
        print(f"ERROR: SFT data not found at {config.SFT_DATA_PATH}")
        print("Run `python generate_sft_data.py` first.")
        sys.exit(1)

    # ── Load student ──────────────────────────────────────────────────────────
    student, tokenizer = load_student(ckpt_path=None, device=device)
    student.train()

    # Enable gradient checkpointing สำหรับประหยัด VRAM
    student.gradient_checkpointing_enable()

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataloader = build_sft_dataloader(
        config.SFT_DATA_PATH,
        tokenizer,
        batch_size=config.SFT_BATCH_SIZE,
    )
    print(f"DataLoader: {len(dataloader)} batches per epoch")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    epochs       = config.SFT_EPOCHS
    grad_accum   = config.SFT_GRAD_ACCUM
    n_total_steps = (len(dataloader) // grad_accum) * epochs
    n_warmup      = int(n_total_steps * config.SFT_WARMUP_RATIO)

    if args.dry_run:
        n_total_steps = args.max_steps * epochs
        n_warmup      = 2

    optimizer = AdamW(student.parameters(), lr=config.SFT_LR, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, n_warmup, n_total_steps)

    print(f"Training: {epochs} epochs, ~{n_total_steps} total optimizer steps")
    print(f"Warmup steps: {n_warmup}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(config.LOG_DIR, "sft_log.jsonl")
    start_time = time.time()

    for epoch in range(epochs):
        metrics = train_epoch(
            model=student,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum=grad_accum,
            max_grad_norm=config.SFT_MAX_GRAD_NORM,
            epoch=epoch,
            max_steps=args.max_steps if args.dry_run else None,
        )

        elapsed = (time.time() - start_time) / 60
        print(f"\nEpoch {epoch+1}/{epochs} — loss: {metrics['loss']:.4f} | time: {elapsed:.1f} min")

        # Log
        import json
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch+1, **metrics, "time_min": elapsed}) + "\n")

        # ── Overfitting guard ──────────────────────────────────────────────────
        if metrics["loss"] < 0.3:
            print("⚠️  Warning: SFT loss very low (<0.3). Risk of overfitting!")
            print("   Consider stopping early. Distillation may not improve much.")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    print(f"\nSaving SFT checkpoint to {config.SFT_CKPT_DIR}...")
    student.save_pretrained(config.SFT_CKPT_DIR)
    tokenizer.save_pretrained(config.SFT_CKPT_DIR)
    print("✅ SFT Phase 1 complete!")
    print(f"   Checkpoint: {config.SFT_CKPT_DIR}")
    print(f"   Final loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: SFT warm-up training")
    parser.add_argument("--dry_run",   action="store_true", help="Quick sanity check")
    parser.add_argument("--max_steps", type=int, default=3,  help="Steps per epoch (dry_run)")
    args = parser.parse_args()
    main(args)
