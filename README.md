# Qwen3.5-0.8B-Disstill-logits-Qwen3.5-9B

2-phase Knowledge Distillation Pipeline จาก Teacher 9B-Instruct สู่ Student 0.8B-Base บน NVIDIA H100 GPU ตีพิมพ์จากแนวทางของ Alibaba Qwen Team.

## Pipeline Overview

1. **Step 1: Data Generation (`generate_sft_data.py`)** 
   - ให้ Teacher 9B สร้าง responses ด้วย **Thinking Mode**
   - ใช้ Mixed Dataset 15,000 samples (70% Math, 20% General, 10% Coding) รันบน inference loop
2. **Phase 1: Warm-up SFT (`train_sft.py`)**
   - Off-policy training
   - สอน Instruction Format ให้ 0.8B-Base และ Warm-up reasoning capacity ก่อน
3. **Phase 2: KL Distillation (`train_distill.py`)**
   - On-policy training
   - Student สร้าง response สดใน loop, Teacher ส่ง soft labels บน tokens เดียวกัน
   - Loss = $\alpha \cdot \text{KL(teacher || student)} + (1-\alpha) \cdot \text{CE}$

## Hardware Requirements

- **GPU**: NVIDIA H100 80GB (หรือเทียบเท่า)
- **VRAM Total Required**: ~36GB
  - Teacher 9B (`bfloat16`): ~18GB
  - Student 0.8B (`float16`): ~2GB + Optimizer/Gradients ~4GB
  - Activations & KV Cache: ~10-12GB

## How to Run

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. End-to-End Execution

รันออโต้ตั้งต้นจนจบ:
```bash
./run_pipeline.sh
```

หรือรันทีละขั้นตอน:
```bash
python generate_sft_data.py
python train_sft.py
python train_distill.py
```

### 3. Quick Test (Dry Run)

เช็คว่าทุกอย่างเวิร์คโดยไม่ต้องรอโหลด dataset เต็มๆ:
```bash
python generate_sft_data.py --dry_run --n_math 5 --n_general 2 --n_coding 2
python train_sft.py --dry_run --max_steps 3
python train_distill.py --dry_run --max_steps 3
```

## Directory Structure

- `data/` : เก็บ `sft_data.jsonl` ที่ generate จาก teacher
- `checkpoints/sft_final/` : จุดเซฟของ Phase 1 (SFT)
- `output/distilled_0.8b/` : โมเดล Final สิ้นสุด Phase 2 (Distilled)
- `logs/` : เก็บ history การเทรน

## References

- **Teacher Model**: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Student Model**: [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
