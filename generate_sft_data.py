"""
generate_sft_data.py — Step 1: Teacher (Qwen3.5-9B) generates responses
Saves {domain, prompt, response} to ./data/sft_data.jsonl

Uses vLLM for high-throughput generation on H100.

Usage:
    python generate_sft_data.py
    python generate_sft_data.py --dry_run --n_math 5 --n_general 2 --n_coding 2
"""

import argparse
import json
import os
import sys
import random

import config
from utils.data_utils import load_prompts
from utils.model_utils import build_chat_prompt

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM is not installed. Please install it with `pip install vllm`.")
    sys.exit(1)


SYSTEM_PROMPTS = {
    "math": (
        "You are an expert mathematician. Solve the given problem step by step, "
        "showing all your reasoning. Be precise and thorough."
    ),
    "general": (
        "You are a helpful, honest, and harmless AI assistant. "
        "Answer the user's question clearly and concisely."
    ),
    "coding": (
        "You are an expert software engineer. Write clean, well-commented code "
        "and explain your solution clearly."
    ),
}

# No need for manual instruction injection, Qwen3.5 natively supports enable_thinking 
# via its chat template.


def main(args):
    # ── Token Authentication ──────────────────────────────────────────────────
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"Failed to log in to Hugging Face Hub: {e}")

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out_path = config.SFT_DATA_PATH

    # ── Load prompts ──────────────────────────────────────────────────────────
    n_math    = args.n_math    if args.dry_run else config.N_MATH
    n_general = args.n_general if args.dry_run else config.N_GENERAL
    n_coding  = args.n_coding  if args.dry_run else config.N_CODING

    prompts = load_prompts(n_math=n_math, n_general=n_general, n_coding=n_coding)
    print(f"Total prompts to generate: {len(prompts)}")

    # Count existing samples (resume support)
    existing = 0
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Found {existing} existing samples. Resuming from sample {existing}.")

    prompts_to_gen = prompts[existing:]
    if not prompts_to_gen:
        print("All prompts already generated. Exiting.")
        return

    # ── Initialize vLLM ───────────────────────────────────────────────────────
    print(f"Loading teacher model ({config.TEACHER_MODEL}) via vLLM...")
    
    # Optional GPU memory utilization config
    # 0.9 = Use 90% of GPU memory for KV cache and model
    gpu_memory_utilization = 0.9 
    
    llm = LLM(
        model=config.TEACHER_MODEL,
        trust_remote_code=True,
        dtype=config.DTYPE,  # usually "bfloat16" for H100
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1, # Change if using multiple GPUs for a single model
        enforce_eager=True,      # Prevent CUDA graph compilation hangs
        max_num_seqs=64,         # Prevent OOM/deadlocks on massive batches
        disable_custom_all_reduce=True,
    )
    tokenizer = llm.get_tokenizer()

    # ── Format Prompts for vLLM ───────────────────────────────────────────────
    formatted_prompts = []
    sampling_params_list = []
    
    # กำหนด random seed สำหรับการแบ่งสัดส่วน thinking
    random.seed(config.SEED)
    
    for item in prompts_to_gen:
        domain = item["domain"]
        sys_p  = SYSTEM_PROMPTS[domain]
        user_p = item["prompt"]
        
        # Data Mixing Strategy
        # Math -> 100% Thinking
        # General/Coding -> 50% Thinking, 50% No-Thinking
        if domain == "math":
            use_thinking = True
        else:
            use_thinking = random.choice([True, False])
            
        item["use_thinking"] = use_thinking
            
        chat_p = build_chat_prompt(tokenizer, sys_p, user_p, enable_thinking=use_thinking)
        formatted_prompts.append(chat_p)
        
        # Select official sampling parameters for this prompt
        s_params_dict = config.THINK_PARAMS if use_thinking else config.NOTHINK_PARAMS
        sampling_params_list.append(
            SamplingParams(
                max_tokens=config.MAX_NEW_TOKENS,
                skip_special_tokens=False, # We want to keep <think> tags if any
                **s_params_dict
            )
        )

    # ── Generation ────────────────────────────────────────────────────────────
    print(f"Starting vLLM generation for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params_list)

    # ── Save Results ──────────────────────────────────────────────────────────
    with open(out_path, "a", encoding="utf-8") as out_f:
        for item, output in zip(prompts_to_gen, outputs):
            # vllm returns the generated text in output.outputs[0].text
            generated_text = output.outputs[0].text
            
            # Clean up ChatML tags if present
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            
            # ตรวจสอบว่ามี <think> tag ออกมาไหม เพื่อเก็บเป็น meta-data
            has_think = "<think>" in generated_text
            
            record = {
                "domain":   item["domain"],
                "prompt":   item["prompt"],
                "response": generated_text,
                "intended_thinking": item.get("use_thinking", True),
                "actual_thinking": has_think
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = existing + len(prompts_to_gen)
    print(f"\nDone! {total} samples saved to {out_path}")

    # Thinking mode sanity check (Check the first generated item)
    if outputs:
        first_gen = outputs[0].outputs[0].text
        has_think = "<think>" in first_gen
        print(f"Thinking tags present in first generated output: {'✅' if has_think else '⚠️ NO/Not Confirmed'}")

    # ── Push Dataset to Hub ───────────────────────────────────────────────────
    if getattr(config, "PUSH_TO_HUB", False) and getattr(config, "HF_DATASET_REPO", "") and not args.dry_run:
        print(f"\nPushing dataset to Hugging Face Hub ({config.HF_DATASET_REPO})...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=config.HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo="sft_data.jsonl",
                repo_id=config.HF_DATASET_REPO,
                repo_type="dataset",
                commit_message="Upload newly generated SFT dataset"
            )
            print("✅ Successfully pushed dataset to Hub!")
        except Exception as e:
            print(f"⚠️ Failed to push dataset to Hub: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data from teacher model using vLLM")
    parser.add_argument("--dry_run",   action="store_true", help="Run with tiny dataset")
    parser.add_argument("--n_math",    type=int, default=5,  help="Math samples (dry_run)")
    parser.add_argument("--n_general", type=int, default=2,  help="General samples (dry_run)")
    parser.add_argument("--n_coding",  type=int, default=2,  help="Coding samples (dry_run)")
    parser.add_argument("--hf_token",  type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()
    main(args)
