"""
utils/distill_loss.py — Loss functions for knowledge distillation
Forward KL: KL(teacher || student) — mean-seeking, stable for warm-up
Reverse KL: KL(student || teacher) — mode-seeking, sharper reasoning
Combined:   alpha * KL + (1-alpha) * CE
"""

import torch
import torch.nn.functional as F
from typing import Optional


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward KL Divergence: KL(teacher_probs || student_log_probs)
    = sum(teacher_probs * (log teacher_probs - log student_probs))

    Mean-seeking: student ครอบคลุม distribution กว้าง
    เหมาะกับ Phase 1 warm-up และ <think> reasoning chains

    Args:
        student_logits: (B, T, V) — student model output
        teacher_logits: (B, T, V) — teacher model output (detached)
        temperature: T scaling สำหรับ soft labels (default 2.0)
        attention_mask: (B, T) optional mask สำหรับ padding tokens

    Returns:
        scalar loss
    """
    T = temperature
    # scaled softmax
    teacher_probs   = F.softmax(teacher_logits / T, dim=-1)       # (B, T, V)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # (B, T, V)

    # KL per token: sum over vocab
    kl_per_token = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1)  # (B, T)

    if attention_mask is not None:
        mask = attention_mask.float()
        kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
    else:
        kl_loss = kl_per_token.mean()

    return kl_loss * (T ** 2)  # temperature compensation


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reverse KL Divergence: KL(student_probs || teacher_log_probs)
    = sum(student_probs * (log student_probs - log teacher_probs))

    Mode-seeking: student ชัดเจน collapse ไปที่ teacher's confident peaks
    ใช้สำหรับ Phase 2 เพื่อ sharpen reasoning (ทดลองได้หลัง Phase 1 stable)
    """
    T = temperature
    student_probs     = F.softmax(student_logits / T, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)

    kl_per_token = (student_probs * (student_probs.log() - teacher_log_probs)).sum(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.float()
        kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
    else:
        kl_loss = kl_per_token.mean()

    return kl_loss * (T ** 2)


def ce_loss_from_labels(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Standard Cross-Entropy loss จาก hard labels
    Shift logits/labels สำหรับ causal LM (predict next token)

    Args:
        student_logits: (B, T, V)
        labels:         (B, T) — -100 สำหรับ masked positions

    Returns:
        scalar loss
    """
    # Shift: predict token t+1 จาก token t
    shift_logits = student_logits[..., :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[..., 1:].contiguous()               # (B, T-1)

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def combined_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    attention_mask: Optional[torch.Tensor] = None,
    use_reverse_kl: bool = False,
) -> dict:
    """
    Combined loss = alpha * KL + (1 - alpha) * CE

    Args:
        student_logits:  (B, T, V)
        teacher_logits:  (B, T, V)  detached จาก teacher
        labels:          (B, T)     hard labels with -100 masks
        alpha:           weight สำหรับ KL term (default 0.5)
        temperature:     KL temperature scaling
        attention_mask:  optional padding mask
        use_reverse_kl:  False=ForwardKL (default), True=ReverseKL

    Returns:
        dict with keys: total_loss, kl_loss, ce_loss
    """
    # Teacher logits ต้อง detach เสมอ (ไม่ backprop ผ่าน teacher)
    teacher_logits = teacher_logits.detach()

    # Align sequence length (student อาจสั้นกว่า teacher ในบาง case)
    seq_len = min(student_logits.size(1), teacher_logits.size(1))
    student_logits_kl = student_logits[:, :seq_len, :]
    teacher_logits_kl = teacher_logits[:, :seq_len, :]

    if use_reverse_kl:
        kl = reverse_kl_loss(student_logits_kl, teacher_logits_kl, temperature, attention_mask)
    else:
        kl = forward_kl_loss(student_logits_kl, teacher_logits_kl, temperature, attention_mask)

    ce = ce_loss_from_labels(student_logits, labels)
    total = alpha * kl + (1.0 - alpha) * ce

    return {
        "total_loss": total,
        "kl_loss":    kl.detach(),
        "ce_loss":    ce.detach(),
    }
