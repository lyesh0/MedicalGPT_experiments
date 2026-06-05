"""Few-shot demo: use a trained RM to score multiple models' answers on the same questions.

Shows per-sample reward scores, per-model averages, and a ranking.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_input(conversations, response_text, tokenizer):
    """Build RM input: conversations as chat template + response as assistant turn."""
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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def score_answer(model, tokenizer, conversations, answer, device, max_length=1280):
    inp = build_input(conversations, answer, tokenizer)
    enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    output = model(**enc)
    return output.logits.reshape(-1)[0].item()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_predictions_as_map(path):
    """Load predictions and index by id."""
    preds = load_jsonl(path)
    return {p["id"]: p["answer"] for p in preds}


def main():
    parser = argparse.ArgumentParser(description="RM few-shot scoring demo")
    parser.add_argument("--model_path", required=True, help="Path to RM model (with adapter)")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL files")
    parser.add_argument("--labels", nargs="+", default=None, help="Model display names")
    parser.add_argument("--eval_data", default="data/experiments/eval/medical_eval_50.jsonl",
                        help="Eval set with questions")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--max_length", type=int, default=1280)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(p).replace("_predictions.jsonl", "") for p in args.predictions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load RM ──
    print(f"Loading RM: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    if not hasattr(model, "prepare_inputs_for_generation"):
        import types
        def _prepare(self, *a, **kw):
            raise NotImplementedError
        model.prepare_inputs_for_generation = types.MethodType(_prepare, model)
    model = PeftModel.from_pretrained(model, args.model_path)

    score_head_path = os.path.join(args.model_path, "score_head.pt")
    if os.path.exists(score_head_path):
        base = model.base_model.model
        if hasattr(base, "score"):
            base.score.load_state_dict(torch.load(score_head_path, map_location="cpu"))

    model.eval()

    # ── Load eval questions ──
    eval_samples = load_jsonl(args.eval_data)[:args.num_samples]

    # ── Load all model predictions ──
    all_preds = {}
    for label, path in zip(args.labels, args.predictions):
        if os.path.exists(path):
            all_preds[label] = load_predictions_as_map(path)
        else:
            print(f"[SKIP] {path} not found")

    # ── Score every (model, sample) pair ──
    results = defaultdict(list)
    per_sample_answers = defaultdict(dict)

    for sample in eval_samples:
        qid = sample["id"]
        conversations = sample.get("conversations", [])
        # Build minimal conversations if not already in chat format
        if not conversations:
            conversations = [{"from": "human", "value": sample["question"]}]

        for label in args.labels:
            if label not in all_preds:
                continue
            answer = all_preds[label].get(qid, "")
            if not answer:
                continue
            score = score_answer(model, tokenizer, conversations, answer, device, args.max_length)
            results[label].append(score)
            per_sample_answers[qid][label] = {"answer": answer[:200], "score": score}

    # ── Print per-sample table ──
    print("\n" + "=" * 100)
    print("RM FEW-SHOT SCORING — PER SAMPLE")
    print("=" * 100)

    header = f"{'ID':<14s} {'Category':<8s} " + " ".join(f"{lbl:>10s}" for lbl in args.labels)
    print(f"\n{header}")
    print("-" * len(header))

    for sample in eval_samples:
        qid = sample["id"]
        cat = sample.get("category", "?")[:8]
        scores = []
        for lbl in args.labels:
            s = per_sample_answers.get(qid, {}).get(lbl, {}).get("score", None)
            if s is not None:
                scores.append(f"{s:>10.3f}")
            else:
                scores.append(f"{'N/A':>10s}")
        print(f"{qid:<14s} {cat:<8s} " + " ".join(scores))

    # ── Per-sample detail (answers side by side) ──
    if args.num_samples <= 5:
        print("\n" + "=" * 100)
        print("SAMPLE ANSWERS (truncated to 200 chars)")
        print("=" * 100)
        for sample in eval_samples:
            qid = sample["id"]
            print(f"\n── [{qid}] {sample.get('category','')} ──")
            print(f"  Q: {sample['question'][:120]}")
            for lbl in args.labels:
                info = per_sample_answers.get(qid, {}).get(lbl, {})
                ans = info.get("answer", "N/A")
                sc = info.get("score", float("nan"))
                print(f"  [{lbl}] score={sc:.3f} | {ans[:150]}")

    # ── Per-model summary ──
    print("\n" + "=" * 100)
    print("RM FEW-SHOT SCORING — MODEL SUMMARY")
    print("=" * 100)
    print(f"{'Model':<22s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Median':>8s} {'N':>5s}")
    print("-" * 65)

    ranked = []
    for lbl in args.labels:
        if lbl not in results:
            continue
        arr = np.array(results[lbl])
        ranked.append((lbl, arr.mean(), arr))
        print(f"{lbl:<22s} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f} {float(np.median(arr)):>8.3f} {len(arr):>5d}")

    # ── Ranking ──
    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print("RANKING (by mean reward)")
    print(f"{'='*60}")
    for i, (lbl, mean, arr) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {lbl:<20s} {mean:.3f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
