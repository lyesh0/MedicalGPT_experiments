"""Evaluate a trained reward model on preference data.

Reports:
- Pairwise accuracy (chosen_reward > rejected_reward)
- Reward distribution stats (mean, std, min, max for chosen/rejected)
- Per-category breakdown
- Length bias check (correlation between response length and reward)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_preference_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data


def build_input(conversations, response_text, tokenizer):
    """Build RM input: conversations + response."""
    messages = []
    if isinstance(conversations, str):
        conversations = json.loads(conversations)

    for turn in conversations:
        role = turn.get("from", "")
        if role == "human":
            messages.append({"role": "user", "content": turn["value"]})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})
        elif role == "system":
            messages.append({"role": "system", "content": turn["value"]})

    messages.append({"role": "assistant", "content": response_text})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


@torch.no_grad()
def score_dataset(model, tokenizer, data, device):
    chosen_rewards = []
    rejected_rewards = []
    chosen_lens = []
    rejected_lens = []
    categories = defaultdict(lambda: {"chosen": [], "rejected": [], "correct": 0, "total": 0})

    for item in tqdm(data, desc="Scoring"):
        conversations = item["conversations"]
        chosen_text = item["chosen"]
        rejected_text = item["rejected"]

        # Build inputs
        chosen_input = build_input(conversations, chosen_text, tokenizer)
        rejected_input = build_input(conversations, rejected_text, tokenizer)

        # Tokenize
        chosen_enc = tokenizer(chosen_input, return_tensors="pt", truncation=True, max_length=1280)
        rejected_enc = tokenizer(rejected_input, return_tensors="pt", truncation=True, max_length=1280)

        chosen_enc = {k: v.to(device) for k, v in chosen_enc.items()}
        rejected_enc = {k: v.to(device) for k, v in rejected_enc.items()}

        # Forward
        chosen_out = model(**chosen_enc)
        rejected_out = model(**rejected_enc)

        chosen_r = chosen_out.logits.reshape(-1)[0].item()
        rejected_r = rejected_out.logits.reshape(-1)[0].item()

        chosen_rewards.append(chosen_r)
        rejected_rewards.append(rejected_r)
        chosen_lens.append(chosen_enc["input_ids"].shape[1])
        rejected_lens.append(rejected_enc["input_ids"].shape[1])

        # Category
        cat = item.get("category", "unknown")
        categories[cat]["chosen"].append(chosen_r)
        categories[cat]["rejected"].append(rejected_r)
        categories[cat]["total"] += 1
        if chosen_r > rejected_r:
            categories[cat]["correct"] += 1

    return chosen_rewards, rejected_rewards, chosen_lens, rejected_lens, categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate a reward model")
    parser.add_argument("--model_path", required=True, help="Path to RM model (with adapter)")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--data", required=True, help="Preference data JSONL")
    parser.add_argument("--output", default=None, help="Output JSON path for detailed scores")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print(f"Loading adapter: {args.model_path}")
    if not hasattr(model, "prepare_inputs_for_generation"):
        import types
        def _prepare_inputs_for_generation(self, *args, **kwargs):
            raise NotImplementedError("prepare_inputs_for_generation is not supported for this model")
        model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)
    model = PeftModel.from_pretrained(model, args.model_path)

    # Load saved score head if present (PEFT doesn't save base model weights)
    score_head_path = os.path.join(args.model_path, "score_head.pt")
    if os.path.exists(score_head_path):
        print(f"Loading score head from {score_head_path}")
        base_model = model.base_model.model
        if hasattr(base_model, "score"):
            base_model.score.load_state_dict(torch.load(score_head_path, map_location="cpu"))
            print(f"Score head loaded, shape: {base_model.score.weight.shape}")

    model.eval()

    data = load_preference_data(args.data)
    print(f"Loaded {len(data)} preference pairs")

    chosen_r, rejected_r, chosen_l, rejected_l, categories = score_dataset(
        model, tokenizer, data, device
    )

    # ── Overall stats ──
    chosen_arr = np.array(chosen_r)
    rejected_arr = np.array(rejected_r)
    accuracy = (chosen_arr > rejected_arr).mean()

    print("\n" + "=" * 60)
    print("REWARD MODEL EVALUATION")
    print("=" * 60)
    print(f"  Pairs evaluated:      {len(chosen_arr)}")
    print(f"  Pairwise accuracy:    {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()

    print("  Chosen reward:")
    print(f"    mean={chosen_arr.mean():.4f}  std={chosen_arr.std():.4f}")
    print(f"    min={chosen_arr.min():.4f}  max={chosen_arr.max():.4f}")
    print()
    print("  Rejected reward:")
    print(f"    mean={rejected_arr.mean():.4f}  std={rejected_arr.std():.4f}")
    print(f"    min={rejected_arr.min():.4f}  max={rejected_arr.max():.4f}")
    print()

    diff = chosen_arr - rejected_arr
    print(f"  Reward diff (chosen - rejected):")
    print(f"    mean={diff.mean():.4f}  std={diff.std():.4f}")
    print(f"    near_zero (< 0.1): {(np.abs(diff) < 0.1).sum()} / {len(diff)}")
    print()

    # ── Length bias check ──
    chosen_len_arr = np.array(chosen_l)
    rejected_len_arr = np.array(rejected_l)
    len_diff = chosen_len_arr - rejected_len_arr

    # Correlation between length difference and reward difference
    mask = len_diff != 0
    if mask.sum() > 5:
        corr, pval = pearsonr(len_diff[mask], diff[mask])
        print(f"  Length bias: r(len_diff, reward_diff) = {corr:.4f} (p={pval:.4f})")
    else:
        print("  Length bias: not computed (too few length-different pairs)")

    # Check if RM simply prefers longer responses
    longer_is_chosen = (chosen_len_arr > rejected_len_arr).mean()
    longer_wins = (diff > 0) & (len_diff > 0)
    longer_wins_ratio = longer_wins.sum() / max((len_diff > 0).sum(), 1)
    print(f"  Chosen longer than rejected: {longer_is_chosen:.2%}")
    print(f"  Longer-chosen pairs where reward agrees: {longer_wins_ratio:.2%}")
    print()

    # ── Per-category breakdown ──
    print("-" * 60)
    print(f"{'Category':<20} {'Count':>6}  {'Accuracy':>10}")
    print("-" * 60)
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {cat:<20} {stats['total']:>5}  {acc:>10.2%}")
    print("-" * 60)

    # ── Acceptance criteria ──
    print()
    if accuracy >= 0.70:
        print("[PASS] Pairwise accuracy >= 70%")
    else:
        print(f"[FAIL] Pairwise accuracy {accuracy:.1%} < 70%")

    near_zero_pct = (np.abs(diff) < 0.1).sum() / len(diff)
    if near_zero_pct < 0.5:
        print(f"[PASS] Near-zero rewards: {near_zero_pct:.1%} < 50%")
    else:
        print(f"[WARN] Near-zero rewards: {near_zero_pct:.1%} >= 50% — rewards may be too similar")

    # Save detailed scores
    if args.output:
        results = {
            "model_path": args.model_path,
            "num_pairs": len(chosen_arr),
            "pairwise_accuracy": float(accuracy),
            "chosen_reward_mean": float(chosen_arr.mean()),
            "chosen_reward_std": float(chosen_arr.std()),
            "rejected_reward_mean": float(rejected_arr.mean()),
            "rejected_reward_std": float(rejected_arr.std()),
            "reward_diff_mean": float(diff.mean()),
            "reward_diff_std": float(diff.std()),
            "near_zero_ratio": float(near_zero_pct),
            "per_category": {
                cat: {
                    "count": stats["total"],
                    "accuracy": stats["correct"] / max(stats["total"], 1),
                    "chosen_mean": float(np.mean(stats["chosen"])),
                    "rejected_mean": float(np.mean(stats["rejected"])),
                }
                for cat, stats in categories.items()
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
