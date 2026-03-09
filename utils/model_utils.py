"""
utils/model_utils.py — Model and tokenizer loading helpers
Teacher: Qwen3.5-9B (bf16, no quantization needed on H100)
Student: Qwen3.5-0.8B-Base (fp16)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_teacher(device: str = "cuda:0") -> tuple:
    """
    Load Qwen3.5-9B as teacher in bfloat16.
    บน H100 80GB: ~18 GB VRAM สบายมาก ไม่ต้องใช้ quantization
    Returns: (model, tokenizer)
    """
    print(f"Loading teacher: {config.TEACHER_MODEL} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.TEACHER_MODEL,
        trust_remote_code=True,
        padding_side="left",   # สำหรับ batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa" if getattr(config, "USE_SDPA", True) else "eager",
    )
    model.eval()
    print(f"  Teacher loaded — params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def load_student(
    ckpt_path: str | None = None,
    device: str = "cuda:0",
) -> tuple:
    """
    Load Qwen3.5-0.8B-Base as student in float16.
    ถ้า ckpt_path=None => โหลด pretrained base จาก HuggingFace
    ถ้า ckpt_path มีค่า => โหลดจาก SFT checkpoint (สำหรับ Phase 2)
    Returns: (model, tokenizer)
    """
    model_id = ckpt_path if ckpt_path else config.STUDENT_MODEL
    print(f"Loading student: {model_id} on {device}")

    # tokenizer เสมอโหลดจาก base model (vocab เหมือนกัน)
    tokenizer = AutoTokenizer.from_pretrained(
        config.STUDENT_MODEL,
        trust_remote_code=True,
        padding_side="right",  # สำหรับ training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # bf16 = H100 native, ไม่ overflow เหมือน fp16
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa" if getattr(config, "USE_SDPA", True) else "eager",
    )
    print(f"  Student loaded — params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


def build_chat_prompt(tokenizer, system_prompt: str, user_message: str, enable_thinking: bool = True) -> str:
    """
    สร้าง chat-formatted prompt สำหรับ teacher inference
    Qwen3.5 ใช้ ChatML format พร้อมเปิด/ปิด thinking mode ผ่าน chat_template_kwargs
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    # apply_chat_template รองรับ enable_thinking ผ่าน kwargs
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": enable_thinking}
    )
    return prompt


def estimate_vram_gb(model) -> float:
    """ประมาณ VRAM ที่ใช้จาก parameter count และ dtype"""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = next(model.parameters()).element_size()
    return (total_params * bytes_per_param) / (1024 ** 3)
