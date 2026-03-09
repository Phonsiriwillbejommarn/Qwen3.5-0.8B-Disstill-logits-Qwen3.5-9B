"""
config.py — Centralized hyperparameters and paths
Qwen3.5-9B (teacher) → Qwen3.5-0.8B-Base (student) distillation pipeline
Hardware: H100 80GB
"""

import os

# ─── Models ───────────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen3.5-9B"
STUDENT_MODEL = "Qwen/Qwen3.5-0.8B-Base"   # Base model → SFT phase สอน format ด้วย

# ─── Dataset sizes ────────────────────────────────────────────────────────────
N_MATH    = 10_000   # OpenR1-Math-220k prompts
N_GENERAL =  3_000   # ShareGPT / Alpaca prompts
N_CODING  =  2_000   # CodeAlpaca prompts
N_TOTAL   = N_MATH + N_GENERAL + N_CODING  # 15,000

# Sampling ratios for mixed DataLoader (must sum to 1.0)
DOMAIN_RATIOS = {
    "math":    0.70,
    "general": 0.20,
    "coding":  0.10,
}

# ─── Data sources (HuggingFace dataset IDs) ───────────────────────────────────
MATH_DATASET    = "open-r1/OpenR1-Math-220k"
GENERAL_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"   # ShareGPT
CODING_DATASET  = "sahil2801/CodeAlpaca-20k"

# ─── Paths & Hub ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
SFT_DATA_PATH   = os.path.join(DATA_DIR, "sft_data.jsonl")
SFT_CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints", "sft_final", "step_50")  # ชี้เป้าไปที่โฟลเดอร์ SFT ที่เซฟไว้
OUTPUT_DIR      = os.path.join(BASE_DIR, "output", "distilled_0.8b")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# ─── Hugging Face Hub & W&B ───────────────────────────────────────────────────
PUSH_TO_HUB     = True
HF_REPO_ID      = "Phonsiri/Qwen3.5-0.8B-Base-Distillation-Qwen3.5-9B"
HF_DISTILL_REPO_ID = "Phonsiri/Qwen3.5-0.8B-Distillation-Phase2" # รีสโปสำหรับผลลัพธ์ Distill เพียวๆ
HF_DATASET_REPO = "Phonsiri/Qwen3.5-Distillation-Dataset"
WANDB_PROJECT   = "qwen3.5-distillation"

# ─── Generation (Teacher) ─────────────────────────────────────────────────────
MAX_NEW_TOKENS      = 8192    # Official rec is up to 32k/81k, 8k is safe for H100 SFT
GEN_BATCH_SIZE      = 4       # teacher inference batch size

# Official Qwen3.5 Sampling Parameters
THINK_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0
}
NOTHINK_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0
}

# ─── SFT Phase 1 ──────────────────────────────────────────────────────────────
SFT_EPOCHS          = 2       # warm-up เท่านั้น อย่า overfit
SFT_BATCH_SIZE      = 2       # per GPU (seq_len 8192)
SFT_GRAD_ACCUM      = 16      # effective batch = 32
SFT_LR              = 2e-5
SFT_LR_SCHEDULER    = "cosine"
SFT_WARMUP_RATIO    = 0.05
SFT_MAX_GRAD_NORM   = 1.0
SFT_LOGGING_STEPS   = 1       # แสดง loss ทุก step เลย
SFT_SAVE_STEPS      = 20      # เซฟเช็คพ้อยท์ทุก 20 steps

# ─── Distillation Phase 2 ─────────────────────────────────────────────────────
DISTILL_EPOCHS       = 3
DISTILL_BATCH_SIZE   = 2       # seq_len 8192
DISTILL_GRAD_ACCUM   = 16      # effective batch = 32
DISTILL_LR           = 1e-5
DISTILL_LR_SCHEDULER = "cosine"
DISTILL_WARMUP_RATIO = 0.05
DISTILL_MAX_GRAD_NORM= 1.0
DISTILL_LOGGING_STEPS= 1       # แสดง loss ทุก step เลย
DISTILL_SAVE_STEPS   = 10      # เซฟเช็คพ้อยท์ทุก 10 steps

# KL distillation settings
KL_TEMPERATURE = 2.0      # temperature scaling สำหรับ soft labels
ALPHA          = 0.5      # KL weight; CE weight = (1 - ALPHA)
TOP_K_LOGITS   = 50       # เก็บแค่ top-K logits เพื่อประหยัด RAM

# ─── Sequence ─────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 7000        # ครอบคลุม 90.4% ของ dataset, สมดุลความเร็ว+คุณภาพ

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED            = 42
DTYPE           = "bfloat16"   # H100 native
USE_SDPA        = True         # Use Scaled Dot Product Attention instead of Flash Attention
