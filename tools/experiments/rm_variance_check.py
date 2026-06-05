"""Check RM score variance across multiple samples from the same question.

Loads SFT on GPU0, RM on GPU1, generates 8 responses per question,
scores them with the RM, and reports per-question variance.
"""

import json
import os
import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM


base_7b = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
rm_path = "outputs/experiments/models/rm_7b_medical"
sft_path = "outputs/experiments/models/sft_7b_C"

dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")

print("Loading SFT on GPU 0...")
tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
sft_model = AutoModelForCausalLM.from_pretrained(
    base_7b, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(dev0)
sft_model = PeftModel.from_pretrained(sft_model, sft_path)
sft_model.eval()

print("Loading RM on GPU 1...")
rm_tokenizer = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    base_7b, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(dev1)
import types
def _prepare(self, *a, **kw):
    raise NotImplementedError
rm_model.prepare_inputs_for_generation = types.MethodType(_prepare, rm_model)
rm_model = PeftModel.from_pretrained(rm_model, rm_path)

score_head_path = os.path.join(rm_path, "score_head.pt")
if os.path.exists(score_head_path):
    base = rm_model.base_model.model
    if hasattr(base, "score"):
        base.score.load_state_dict(torch.load(score_head_path, map_location="cpu"))
rm_model.eval()
print("Both models loaded.")


@torch.no_grad()
def generate(prompt, temp):
    msgs = [{"role": "user", "content": prompt}]
    inp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=512).to(dev0)
    out = sft_model.generate(
        **enc,
        max_new_tokens=256,
        temperature=temp,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def rm_score(conversations, answer):
    msgs = []
    for turn in conversations:
        role = turn.get("from", "")
        if role == "human":
            msgs.append({"role": "user", "content": turn["value"]})
        elif role == "gpt":
            msgs.append({"role": "assistant", "content": turn["value"]})
    msgs.append({"role": "assistant", "content": answer})
    text = rm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=1280)
    enc = {k: v.to(dev1) for k, v in enc.items()}
    return rm_model(**enc).logits.reshape(-1)[0].item()


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main():
    eval_data = load_jsonl("data/experiments/eval/medical_eval_50.jsonl")[:5]
    n_per_q = 8

    print()
    print("=" * 80)
    print("VARIANCE CHECK: sft_7b_C -> 7B RM")
    print("=" * 80)

    all_q_results = []

    for sample in eval_data:
        qid = sample["id"]
        cat = sample.get("category", "?")
        question = sample["question"]
        conversations = sample.get("conversations", [])
        if not conversations:
            conversations = [{"from": "human", "value": question}]

        print(f"\n-- [{qid}] {cat} --")
        print(f"  Q: {question[:100]}")

        scores = []
        for i in range(n_per_q):
            temp = 0.3 + i * 0.15 if i > 0 else 0.1
            ans = generate(question, temp)
            sc = rm_score(conversations, ans)
            scores.append(sc)
            if i < 2 or i >= n_per_q - 2:
                print(f"  [{i}] T={temp:.2f} score={sc:.3f} | {ans[:60]}...")
            elif i == 2:
                print(f"  ...")

        arr = np.array(scores)
        all_q_results.append(
            {"qid": qid, "mean": float(arr.mean()), "std": float(arr.std()),
             "min": float(arr.min()), "max": float(arr.max())}
        )
        print(f"  => mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}  range={arr.max()-arr.min():.3f}")

        if arr.std() < 0.1:
            print(f"     DANGER: std<0.1")
        elif arr.std() < 0.3:
            print(f"     MARGINAL")
        else:
            print(f"     GOOD")

    print()
    print("=" * 80)
    print("Summary:")
    for r in all_q_results:
        print(f"  {r['qid']}: mean={r['mean']:.3f}  std={r['std']:.3f}  range={r['max']-r['min']:.3f}")
    avg_std = np.mean([r["std"] for r in all_q_results])
    print(f"\n  Avg std = {avg_std:.3f}")

    if avg_std > 0.3:
        print("  => RM has within-distribution discrimination, RLOO viable")
    elif avg_std > 0.1:
        print("  => Weak discrimination, RLOO may converge slowly - increase temperature or num_generations")
    else:
        print("  => No discrimination, RLOO will fail even with good pairwise accuracy")

    print("\nDone.")


if __name__ == "__main__":
    main()
