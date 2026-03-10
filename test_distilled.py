from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# ── Load model ───────────────────────────────────────────────
MODEL_ID  = "Phonsiri/Qwen3.5-0.8B-Distillation-Phase2"
SUBFOLDER = "epoch_1_step_50" # เปลี่ยนเลขให้ตรงกับล่าสุดของคุณได้เลยครับ

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, subfolder=SUBFOLDER, trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    subfolder=SUBFOLDER,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("✅ Model loaded!")

# ── Inference helper (streaming) ─────────────────────────────
def chat(user_prompt: str, system_prompt: str = None,
         enable_thinking: bool = True, max_new_tokens: int = 1000):

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # TextStreamer พิมพ์ token ทีละตัวแบบ real-time
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,           # ไม่แสดง prompt ซ้ำ
        skip_special_tokens=False,  # เก็บ <think> tags ไว้ เพื่อดูว่ามันวิเคราะห์มั้ย
    )

    print(f"\n[USER]: {user_prompt}\n")
    print("[BOT]: ", end="")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.05, 
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id, 
        )


# ── Test prompts ─────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 — Math (Thinking ON)")
print("="*60)
chat(
    user_prompt="A hemisphere with radius $200$ sits on top of a horizontal circular disk with radius $200,$ and the hemisphere and disk have the same center. Let $\mathcal T$ be the region of points P in the disk such that a sphere of radius $42$ can be placed on top of the disk at $P$ and lie completely inside the hemisphere. The area of $\mathcal T$ divided by the area of the disk is $\tfrac pq,$ where $p$ and $q$ are relatively prime positive integers. Find $p+q.$",
    system_prompt="You are an expert mathematician. Solve step by step.",
    enable_thinking=True,
    max_new_tokens=1500, # คณิตศาสตร์อาจคิดยาว เลยเผื่อไว้ 1500
)

print("\n" + "="*60)
print("TEST 2 — General (Thinking OFF)")
print("="*60)
chat(
    user_prompt="What is knowledge distillation in machine learning?",
    system_prompt="You are a helpful AI assistant.",
    enable_thinking=False,
    max_new_tokens=800,
)

print("\n" + "="*60)
print("TEST 3 — Coding (Thinking ON)")
print("="*60)
chat(
    user_prompt="Write a Python function to compute the Fibonacci sequence up to n terms.",
    system_prompt="You are an expert software engineer.",
    enable_thinking=True,
    max_new_tokens=1000,
)
