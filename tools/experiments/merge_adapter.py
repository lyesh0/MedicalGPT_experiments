"""Merge LoRA adapter into base model and save to disk."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

print(f"Loading base model: {args.base_model}")
model = AutoModelForCausalLM.from_pretrained(
    args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
)
print(f"Loading adapter: {args.adapter}")
model = PeftModel.from_pretrained(model, args.adapter)
print("Merging...")
model = model.merge_and_unload()
print(f"Saving merged model to: {args.output}")
model.save_pretrained(args.output)

tokenizer = AutoTokenizer.from_pretrained(args.adapter)
tokenizer.save_pretrained(args.output)
print("Done.")
