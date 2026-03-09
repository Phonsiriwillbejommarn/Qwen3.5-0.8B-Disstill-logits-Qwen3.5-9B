"""
train_distill.py — Phase 2: On-policy Knowledge Distillation with Forward KL
Student (from SFT checkpoint) aligns logits with teacher (9B-Instruct)

Flow per batch:
  1. Generate student response (on-policy)
  2. Teacher forward pass on same tokens → soft labels
  3. Loss = alpha * ForwardKL(teacher || student) + (1-alpha) * CE
  4. Backprop through student only (teacher frozen)

Usage:
    python train_distill.py
    python train_distill.py --dry_run --max_steps 3
    python train_distill.py --use_reverse_kl   # ทดลอง Reverse KL ใน Phase 2
"""

import argparse
import os
import sys
import time
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi

import config
from utils.model_utils import load_teacher, load_student
from utils.data_utils import build_distill_dataloader
from utils.distill_loss import combined_distill_loss


def get_scheduler(optimizer, num_warmup: int, num_total: int):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=num_total - num_warmup, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[num_warmup])


@torch.no_grad()
def teacher_forward(teacher, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Teacher forward pass — no grad, returns logits
    teacher ต้อง eval mode และ frozen เสมอ
    """
    out = teacher(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits  # (B, T, V)


def generate_student_response(
    student,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    On-policy: student generates response autoregressively
    Returns: (full_input_ids, full_attention_mask) including prompt + response
    """
    student.eval()
    with torch.no_grad():
        gen_out = student.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    student.train()

    # Pad gen_out ถ้าสั้นกว่า prompt (ไม่น่าเกิด แต่defensive)
    full_mask = torch.ones_like(gen_out)
    if tokenizer.pad_token_id is not None:
        full_mask = (gen_out != tokenizer.pad_token_id).long()

    return gen_out, full_mask


def build_labels(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mask prompt positions with -100, train only on generated response"""
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return labels


def train_step(
    student,
    teacher,
    tokenizer,
    batch: dict,
    optimizer,
    scheduler,
    grad_accum: int,
    batch_idx: int,
    args,
    device: str,
) -> dict:
    """Single training step with KL + CE loss"""

    prompt_ids  = batch["input_ids"].to(device)    # (B, T_prompt)
    prompt_mask = batch["attention_mask"].to(device)
    prompt_len  = prompt_ids.shape[1]

    # Step 1: Student generates on-policy response
    full_ids, full_mask = generate_student_response(
        student, tokenizer, prompt_ids, prompt_mask,
        max_new=min(512, config.MAX_SEQ_LEN - prompt_len),
    )
    full_ids  = full_ids.to(device)
    full_mask = full_mask.to(device)

    # Step 2: Teacher forward (frozen, no_grad)
    with torch.no_grad():
        teacher_logits = teacher_forward(teacher, full_ids, full_mask)  # (B, T, V)

    # Step 3: Student forward (trainable)
    student_out    = student(input_ids=full_ids, attention_mask=full_mask)
    student_logits = student_out.logits  # (B, T, V)

    # Step 4: Build labels (mask prompt portion)
    labels = build_labels(full_ids, prompt_len)

    # Step 5: Compute combined loss
    loss_dict = combined_distill_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        alpha=config.ALPHA,
        temperature=config.KL_TEMPERATURE,
        attention_mask=full_mask,
        use_reverse_kl=args.use_reverse_kl,
    )

    total_loss = loss_dict["total_loss"] / grad_accum
    total_loss.backward()

    # Gradient step
    did_step = False
    if (batch_idx + 1) % grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.DISTILL_MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        did_step = True

    return {
        "total_loss": loss_dict["total_loss"].item(),
        "kl_loss":    loss_dict["kl_loss"].item(),
        "ce_loss":    loss_dict["ce_loss"].item(),
        "did_step":   did_step,
        "seq_len":    full_ids.shape[1],
    }


def main(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(config.SEED)
    print(f"Device: {device}")
    print(f"KL type: {'Reverse KL' if args.use_reverse_kl else 'Forward KL (default)'}")

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

    # ── Check SFT checkpoint ──────────────────────────────────────────────────
    if not os.path.exists(config.SFT_CKPT_DIR):
        print(f"ERROR: SFT checkpoint not found at {config.SFT_CKPT_DIR}")
        print("Run `python train_sft.py` first.")
        sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────────────
    # Teacher on cuda:0, Student on cuda:0 (H100 80GB พอสำหรับทั้งคู่)
    # ถ้า VRAM ตึง: teacher=cuda:0, student=cuda:1
    teacher, t_tokenizer = load_teacher(device=device)
    student, s_tokenizer = load_student(ckpt_path=config.SFT_CKPT_DIR, device=device)

    teacher.eval()   # teacher frozen ตลอด
    student.train()
    student.gradient_checkpointing_enable()

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataloader = build_distill_dataloader(
        config.SFT_DATA_PATH, s_tokenizer, batch_size=config.DISTILL_BATCH_SIZE
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    epochs       = config.DISTILL_EPOCHS
    grad_accum   = config.DISTILL_GRAD_ACCUM
    n_total_steps = (len(dataloader) // grad_accum) * epochs
    n_warmup      = int(n_total_steps * config.DISTILL_WARMUP_RATIO)

    if args.dry_run:
        n_total_steps = args.max_steps * epochs
        n_warmup      = 1

    optimizer = AdamW(student.parameters(), lr=config.DISTILL_LR, weight_decay=0.01)
    scheduler = get_scheduler(optimizer, n_warmup, n_total_steps)

    print(f"Training: {epochs} epochs, ~{n_total_steps} optimizer steps, warmup={n_warmup}")

    # ── Init W&B ──────────────────────────────────────────────────────────────
    if not args.dry_run and config.WANDB_PROJECT:
        wandb.init(
            project=config.WANDB_PROJECT,
            name="Phase2-Distillation",
            config={
                "epochs": config.DISTILL_EPOCHS,
                "batch_size": config.DISTILL_BATCH_SIZE * config.DISTILL_GRAD_ACCUM,
                "lr": config.DISTILL_LR,
                "kl_temp": config.KL_TEMPERATURE,
                "alpha": config.ALPHA
            }
        )

    # ── Training Loop ─────────────────────────────────────────────────────────
    log_path   = os.path.join(config.LOG_DIR, "distill_log.jsonl")
    start_time = time.time()
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_total  = 0.0
        epoch_kl     = 0.0
        epoch_ce     = 0.0
        n_steps      = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{epochs} Distill", dynamic_ncols=True)

        for batch_idx, batch in pbar:
            metrics = train_step(
                student=student,
                teacher=teacher,
                tokenizer=s_tokenizer,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_accum=grad_accum,
                batch_idx=batch_idx,
                args=args,
                device=device,
            )

            epoch_total += metrics["total_loss"]
            epoch_kl    += metrics["kl_loss"]
            epoch_ce    += metrics["ce_loss"]
            n_steps     += 1

            if n_steps % config.DISTILL_LOGGING_STEPS == 0:
                avg = epoch_total / n_steps
                lr  = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{avg:.3f}",
                    "kl":   f"{epoch_kl/n_steps:.3f}",
                    "ce":   f"{epoch_ce/n_steps:.3f}",
                    "lr":   f"{lr:.2e}",
                })
                
                # W&B Logging
                if wandb.run is not None:
                    wandb.log({
                        "train/total_loss": avg,
                        "train/kl_loss": epoch_kl/n_steps,
                        "train/ce_loss": epoch_ce/n_steps,
                        "train/lr": lr,
                        "train/step": n_steps,
                        "train/epoch": epoch + (batch_idx / len(dataloader))
                    }, step=n_steps)

            # Step-based checkpoint saving
            if n_steps == 1 or n_steps % config.DISTILL_SAVE_STEPS == 0:
                ckpt_dir = os.path.join(config.OUTPUT_DIR, f"step_{n_steps}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student.save_pretrained(ckpt_dir)
                s_tokenizer.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                print(f"\n💾 Saved checkpoint (Model + Optimizer) step {n_steps} → {ckpt_dir}")

                if config.PUSH_TO_HUB and not args.dry_run:
                    try:
                        api = HfApi()
                        api.create_repo(repo_id=config.HF_DISTILL_REPO_ID, repo_type="model", exist_ok=True)
                        api.upload_folder(
                            folder_path=ckpt_dir,
                            repo_id=config.HF_DISTILL_REPO_ID,
                            path_in_repo=f"distill_step_{n_steps}",
                            repo_type="model",
                            commit_message=f"Distill checkpoint step {n_steps} (Model + Optimizer)"
                        )
                        print(f"  ✅ Pushed distill step {n_steps} to Hub!")
                    except Exception as e:
                        print(f"  ⚠️ Hub push failed: {e}")

            if args.dry_run and n_steps >= args.max_steps:
                break

        elapsed = (time.time() - start_time) / 60
        epoch_metrics = {
            "epoch":     epoch + 1,
            "total":     epoch_total / max(n_steps, 1),
            "kl":        epoch_kl    / max(n_steps, 1),
            "ce":        epoch_ce    / max(n_steps, 1),
            "time_min":  elapsed,
        }
        print(f"\nEpoch {epoch+1} — total: {epoch_metrics['total']:.4f} | "
              f"kl: {epoch_metrics['kl']:.4f} | ce: {epoch_metrics['ce']:.4f} | "
              f"time: {elapsed:.1f}min")

        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")

        # ── Intermediate save ────────────────────────────────────────────────
        epoch_ckpt = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch+1}")
        student.save_pretrained(epoch_ckpt)
        s_tokenizer.save_pretrained(epoch_ckpt)
        torch.save(optimizer.state_dict(), os.path.join(epoch_ckpt, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(epoch_ckpt, "scheduler.pt"))
        print(f"  Saved epoch checkpoint (Model + Optimizer) → {epoch_ckpt}")
        
        if config.PUSH_TO_HUB and not args.dry_run:
            print(f"  Pushing epoch {epoch+1} to Hub...")
            try:
                api = HfApi()
                api.create_repo(repo_id=config.HF_DISTILL_REPO_ID, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=epoch_ckpt,
                    repo_id=config.HF_DISTILL_REPO_ID,
                    path_in_repo=f"epoch_{epoch+1}",
                    repo_type="model",
                    commit_message=f"Upload Distillation Checkpoint Epoch {epoch+1} (Model + Optimizer)"
                )
                print(f"  ✅ Uploaded epoch {epoch+1} to Hub!")
            except Exception as e:
                print(f"  ⚠️ Failed to upload epoch {epoch+1} to Hub: {e}")

    # ── Final save ─────────────────────────────────────────────────────────────
    student.save_pretrained(config.OUTPUT_DIR)
    s_tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(config.OUTPUT_DIR, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(config.OUTPUT_DIR, "scheduler.pt"))
    
    # ── Push to Hub ────────────────────────────────────────────────────────────
    if config.PUSH_TO_HUB and not args.dry_run:
        print(f"\nPushing Distilled Model to Hugging Face Hub ({config.HF_DISTILL_REPO_ID})...")
        try:
            api = HfApi()
            api.create_repo(repo_id=config.HF_DISTILL_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=config.OUTPUT_DIR,
                repo_id=config.HF_DISTILL_REPO_ID,
                repo_type="model",
                commit_message="Upload Final Distilled Checkpoint including Optimizer"
            )
            print("✅ Successfully pushed distilled model to Hub!")
        except Exception as e:
            print(f"⚠️ Failed to push to Hub: {e}")

    print(f"\n✅ Distillation Phase 2 complete!")
    print(f"   Final model: {config.OUTPUT_DIR}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: KL logits distillation")
    parser.add_argument("--dry_run",        action="store_true", help="Quick sanity check")
    parser.add_argument("--max_steps",      type=int, default=3, help="Steps per epoch (dry_run)")
    parser.add_argument("--use_reverse_kl", action="store_true", help="Use Reverse KL instead of Forward KL")
    parser.add_argument("--hf_token",       type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--wandb_key",      type=str, default=None, help="Weights & Biases API key")
    args = parser.parse_args()
    main(args)
