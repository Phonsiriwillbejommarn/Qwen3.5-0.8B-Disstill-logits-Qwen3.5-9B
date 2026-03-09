"""
utils/data_utils.py — Dataset loading and preprocessing utilities
Handles 3-domain mixed dataset: math / general / coding
"""

import os
import json
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import PreTrainedTokenizer
import torch
from typing import Optional


# ─── Prompt Extractors ────────────────────────────────────────────────────────

def extract_math_prompt(example: dict) -> str:
    """Extract question text from OpenR1-Math-220k"""
    # Field name: 'problem' หรือ 'question'
    return example.get("problem") or example.get("question", "")


def extract_general_prompt(example: dict) -> str:
    """Extract first human turn from ShareGPT conversation"""
    conversations = example.get("conversations", [])
    for turn in conversations:
        if turn.get("from") in ("human", "user"):
            return turn.get("value", "").strip()
    return ""


def extract_coding_prompt(example: dict) -> str:
    """Extract instruction from CodeAlpaca"""
    instruction = example.get("instruction", "")
    inp = example.get("input", "").strip()
    return f"{instruction}\n{inp}".strip() if inp else instruction


# ─── Prompt Loading ────────────────────────────────────────────────────────────

def load_prompts(
    n_math: int = config.N_MATH,
    n_general: int = config.N_GENERAL,
    n_coding: int = config.N_CODING,
    seed: int = config.SEED,
) -> list[dict]:
    """
    โหลด prompts จาก 3 domains และ shuffle รวมกัน
    Returns: list of dicts with keys: {domain, prompt}
    """
    from datasets import load_dataset

    random.seed(seed)
    prompts = []

    # ── Math ──────────────────────────────────────────────────────────────────
    print(f"Loading math ({n_math} samples) from {config.MATH_DATASET}...")
    ds_math = load_dataset(config.MATH_DATASET, split="train", streaming=True)
    count = 0
    for ex in ds_math:
        text = extract_math_prompt(ex)
        if text:
            prompts.append({"domain": "math", "prompt": text})
            count += 1
        if count >= n_math:
            break

    # ── General ───────────────────────────────────────────────────────────────
    print(f"Loading general ({n_general} samples) from {config.GENERAL_DATASET}...")
    try:
        ds_gen = load_dataset(config.GENERAL_DATASET, split="train", streaming=True)
        count = 0
        for ex in ds_gen:
            text = extract_general_prompt(ex)
            if text and len(text) > 20:   # กรอง prompts สั้นเกินไป
                prompts.append({"domain": "general", "prompt": text})
                count += 1
            if count >= n_general:
                break
    except Exception as e:
        print(f"  [Warning] ShareGPT load failed: {e}. Using Alpaca fallback...")
        ds_gen = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        for ex in ds_gen:
            text = ex.get("instruction", "")
            if text:
                prompts.append({"domain": "general", "prompt": text})
                count += 1
            if count >= n_general:
                break

    # ── Coding ────────────────────────────────────────────────────────────────
    print(f"Loading coding ({n_coding} samples) from {config.CODING_DATASET}...")
    ds_code = load_dataset(config.CODING_DATASET, split="train", streaming=True)
    count = 0
    for ex in ds_code:
        text = extract_coding_prompt(ex)
        if text:
            prompts.append({"domain": "coding", "prompt": text})
            count += 1
        if count >= n_coding:
            break

    random.shuffle(prompts)
    print(f"Total prompts loaded: {len(prompts)}")
    return prompts


# ─── PT Dataset Classes ────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Phase 1 Dataset: (prompt + teacher response) → SFT training
    โหลดจาก sft_data.jsonl ที่ generate_sft_data.py สร้างไว้

    Format ของแต่ละ sample:
    {
        "domain":   "math" | "general" | "coding",
        "prompt":   str,
        "response": str  (รวม <think>...</think> tags ถ้า thinking_mode=True)
    }
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = config.MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = []

        print(f"Loading SFT data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"  Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt   = sample["prompt"]
        response = sample["response"]

        # สำหรับ Base model ต้องสร้าง full text รวม prompt+response
        # ใช้ ChatML format ให้ student เรียน format ด้วย
        full_text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Labels: mask prompt tokens ด้วย -100, train เฉพาะ response
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_len  = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_len
        )["input_ids"]
        prompt_len  = len(prompt_len)

        labels = input_ids.clone()
        labels[:prompt_len] = -100   # mask prompt — ไม่ train บน prompt

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "domain":         sample["domain"],
        }


class DistillDataset(Dataset):
    """
    Phase 2 Dataset: same as SFT data แต่ไม่มี teacher logits pre-saved
    เพราะทำ on-policy: student generate ก่อน แล้วผ่าน teacher forward pass
    ดังนั้น Dataset นี้แค่เก็บ input_ids ให้ training loop เอาไปใช้

    Note: teacher logits คำนวณ on-the-fly ใน train_distill.py
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = config.MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    self.samples.append(obj["prompt"])  # แค่ prompt สำหรับ on-policy gen

        print(f"DistillDataset loaded {len(self.samples)} prompts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        prompt = self.samples[idx]
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_len // 2,   # เหลือที่ว่างสำหรับ response generation
            return_tensors="pt",
        )
        return {
            "input_ids":      tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }


# ─── Collate Functions ────────────────────────────────────────────────────────

def sft_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad batch to same length สำหรับ SFT training"""
    input_ids      = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels         = [x["labels"] for x in batch]

    max_len = max(t.size(0) for t in input_ids)

    padded_input   = torch.zeros(len(batch), max_len, dtype=torch.long).fill_(pad_token_id)
    padded_mask    = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels  = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, mask, lbl) in enumerate(zip(input_ids, attention_mask, labels)):
        L = ids.size(0)
        padded_input[i, :L]  = ids
        padded_mask[i, :L]   = mask
        padded_labels[i, :L] = lbl

    return {
        "input_ids":      padded_input,
        "attention_mask": padded_mask,
        "labels":         padded_labels,
    }


def distill_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad batch สำหรับ distillation (prompt only)"""
    input_ids      = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    max_len = max(t.size(0) for t in input_ids)

    padded_input = torch.zeros(len(batch), max_len, dtype=torch.long).fill_(pad_token_id)
    padded_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        L = ids.size(0)
        padded_input[i, :L] = ids
        padded_mask[i, :L]  = mask

    return {
        "input_ids":      padded_input,
        "attention_mask": padded_mask,
    }


def build_sft_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = config.SFT_BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    dataset = SFTDataset(data_path, tokenizer)
    pad_id  = tokenizer.pad_token_id

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: sft_collate_fn(b, pad_id),
        pin_memory=True,
    )


def build_distill_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = config.DISTILL_BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    dataset = DistillDataset(data_path, tokenizer)
    pad_id  = tokenizer.pad_token_id

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: distill_collate_fn(b, pad_id),
        pin_memory=True,
    )
