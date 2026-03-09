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
import wandb
from huggingface_hub import HfApi

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
    tokenizer=None,
    save_steps: int = 50,
    push_to_hub: bool = False,
    dry_run: bool = False,
) -> dict:
    """Train one epoch, return metrics"""
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    step = 0
    nan_count = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} SFT", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Skip batches where all labels are masked
        if (labels == -100).all():
            continue

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum

        # NaN detection — skip bad batches
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            if nan_count % 10 == 0:
                print(f"\n⚠️  NaN/Inf loss detected {nan_count} times (skipping batch {batch_idx})")
            continue

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
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "nan": nan_count})
                
                # W&B Logging
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": step,
                        "train/epoch": epoch + (batch_idx / len(dataloader)),
                        "train/nan_count": nan_count,
                    }, step=step)

            # Step-based checkpoint saving
            if step % save_steps == 0 and tokenizer is not None:
                ckpt_dir = os.path.join(config.SFT_CKPT_DIR, f"step_{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"\n💾 Saved checkpoint step {step} → {ckpt_dir}")

                if push_to_hub and not dry_run:
                    try:
                        api = HfApi()
                        api.create_repo(repo_id=config.HF_REPO_ID, repo_type="model", exist_ok=True)
                        api.upload_folder(
                            folder_path=ckpt_dir,
                            repo_id=config.HF_REPO_ID,
                            path_in_repo=f"sft_step_{step}",
                            repo_type="model",
                            commit_message=f"SFT checkpoint step {step}"
                        )
                        print(f"  ✅ Pushed step {step} to Hub!")
                    except Exception as e:
                        print(f"  ⚠️ Hub push failed: {e}")

            if max_steps and step >= max_steps:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    if nan_count > 0:
        print(f"\n⚠️  Total NaN/Inf batches skipped: {nan_count}")
    return {"loss": avg_loss, "steps": step, "nan_count": nan_count}


def main(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(config.SFT_CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(config.SEED)

    # ── Token Authentication ──────────────────────────────────────────────────
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"Failed to log in to Hugging Face Hub: {e}")

    if args.wandb_key:
        try:
            import wandb
            wandb.login(key=args.wandb_key)
            print("Successfully logged in to Weights & Biases.")
        except Exception as e:
            print(f"Failed to log in to Weights & Biases: {e}")

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

    # ── Init W&B ──────────────────────────────────────────────────────────────
    if not args.dry_run and config.WANDB_PROJECT:
        wandb.init(
            project=config.WANDB_PROJECT,
            name="Phase1-SFT",
            config={
                "epochs": config.SFT_EPOCHS,
                "batch_size": config.SFT_BATCH_SIZE * config.SFT_GRAD_ACCUM,
                "lr": config.SFT_LR,
            }
        )

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
            tokenizer=tokenizer,
            save_steps=config.SFT_SAVE_STEPS,
            push_to_hub=config.PUSH_TO_HUB,
            dry_run=args.dry_run,
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

        # ── Intermediate save ────────────────────────────────────────────────
        epoch_ckpt = os.path.join(config.SFT_CKPT_DIR, f"epoch_{epoch+1}")
        student.save_pretrained(epoch_ckpt)
        tokenizer.save_pretrained(epoch_ckpt)
        print(f"  Saved SFT epoch checkpoint → {epoch_ckpt}")
        
        if config.PUSH_TO_HUB and not args.dry_run:
            print(f"  Pushing SFT epoch {epoch+1} to Hub...")
            try:
                api = HfApi()
                api.create_repo(repo_id=config.HF_REPO_ID, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=epoch_ckpt,
                    repo_id=config.HF_REPO_ID,
                    path_in_repo=f"sft_epoch_{epoch+1}",
                    repo_type="model",
                    commit_message=f"Upload SFT Checkpoint Epoch {epoch+1}"
                )
                print(f"  ✅ Uploaded SFT epoch {epoch+1} to Hub!")
            except Exception as e:
                print(f"  ⚠️ Failed to upload SFT epoch {epoch+1} to Hub: {e}")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    print(f"\nSaving SFT checkpoint to {config.SFT_CKPT_DIR}...")
    student.save_pretrained(config.SFT_CKPT_DIR)
    tokenizer.save_pretrained(config.SFT_CKPT_DIR)
    
    # Save optimizer and scheduler for resuming / full checkpoint push
    torch.save(optimizer.state_dict(), os.path.join(config.SFT_CKPT_DIR, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(config.SFT_CKPT_DIR, "scheduler.pt"))
    
    # ── Push to Hub ────────────────────────────────────────────────────────────
    if config.PUSH_TO_HUB and not args.dry_run:
        print(f"Pushing SFT Checkpoint to Hugging Face Hub ({config.HF_REPO_ID})...")
        try:
            api = HfApi()
            api.create_repo(repo_id=config.HF_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=config.SFT_CKPT_DIR,
                path_in_repo="sft_checkpoint",
                repo_id=config.HF_REPO_ID,
                repo_type="model",
                commit_message="Upload SFT Phase 1 Checkpoint including Optimizer"
            )
            print("✅ Successfully pushed SFT checkpoint to Hub!")
        except Exception as e:
            print(f"⚠️ Failed to push to Hub: {e}")

    print("✅ SFT Phase 1 complete!")
    print(f"   Checkpoint: {config.SFT_CKPT_DIR}")
    print(f"   Final loss: {metrics['loss']:.4f}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: SFT warm-up training")
    parser.add_argument("--dry_run",   action="store_true", help="Quick sanity check")
    parser.add_argument("--max_steps", type=int, default=3,  help="Steps per epoch (dry_run)")
    parser.add_argument("--hf_token",  type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API key")
    args = parser.parse_args()
    main(args)
