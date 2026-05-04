#!/usr/bin/env python3
"""
Batch inference: load base model + LoRA adapter, run inference on eval set.

Usage:
    python tools/experiments/batch_inference.py \
        --base_model Qwen/Qwen3.5-2B \
        --lora_model outputs/experiments/models/sft_A \
        --eval_file data/experiments/eval/medical_eval_50.jsonl \
        --output outputs/experiments/predictions/sft_A_predictions.jsonl

    # Or compare multiple models at once:
    python tools/experiments/batch_inference.py \
        --base_model Qwen/Qwen3.5-2B \
        --lora_models outputs/experiments/models/sft_A \
                      outputs/experiments/models/sft_B \
                      outputs/experiments/models/sft_C \
        --eval_file data/experiments/eval/medical_eval_50.jsonl
"""

import json
import os
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_jsonl(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def save_jsonl(rows: list[dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(base_model: str, lora_model: str | None, device: str = "cuda"):
    """Load base model with optional LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_model and os.path.isdir(lora_model):
        print(f"Loading LoRA adapter: {lora_model}")
        model = PeftModel.from_pretrained(model, lora_model)
        model = model.merge_and_unload()
        print("  LoRA merged")

    model.eval()
    return model, tokenizer


def format_prompt(question: str, tokenizer) -> str:
    """Format a single question as a chat prompt."""
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.1, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(model, tokenizer, eval_rows: list[dict],
                  max_new_tokens: int, temperature: float) -> list[dict]:
    results = []
    for i, item in enumerate(eval_rows):
        prompt = format_prompt(item["question"], tokenizer)
        answer = generate(model, tokenizer, prompt, max_new_tokens, temperature)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(eval_rows)}")

    return results


def infer_one_model(base_model: str, lora_model: str | None, eval_rows: list[dict],
                    output: str, max_new_tokens: int, temperature: float):
    model_name = os.path.basename(lora_model) if lora_model else os.path.basename(base_model)
    print(f"\n{'='*50}")
    print(f"Inference: {model_name}")
    print(f"{'='*50}")

    model, tokenizer = load_model_and_tokenizer(base_model, lora_model)
    results = run_inference(model, tokenizer, eval_rows, max_new_tokens, temperature)
    save_jsonl(results, output)
    print(f"  Saved {len(results)} predictions to {output}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Batch inference for SFT model comparison")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--lora_model", type=str, default=None,
                        help="Single LoRA adapter path")
    parser.add_argument("--lora_models", nargs="+", default=None,
                        help="Multiple LoRA adapter paths")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Evaluation set JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (used with --lora_model)")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments/predictions",
                        help="Output directory (used with --lora_models)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    eval_rows = load_jsonl(args.eval_file)
    if not eval_rows:
        print(f"No data in {args.eval_file}")
        sys.exit(1)
    print(f"Loaded {len(eval_rows)} eval samples")

    # Single model mode
    if args.lora_model:
        output = args.output or os.path.join(
            args.output_dir, f"{os.path.basename(args.lora_model)}_predictions.jsonl"
        )
        infer_one_model(args.base_model, args.lora_model, eval_rows,
                        output, args.max_new_tokens, args.temperature)
        return

    # Multi-model mode
    if args.lora_models:
        for lora_path in args.lora_models:
            name = os.path.basename(lora_path.rstrip("/"))
            output = os.path.join(args.output_dir, f"{name}_predictions.jsonl")
            infer_one_model(args.base_model, lora_path, eval_rows,
                            output, args.max_new_tokens, args.temperature)
        return

    print("Specify --lora_model or --lora_models")
    sys.exit(1)


if __name__ == "__main__":
    main()
