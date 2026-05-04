#!/usr/bin/env python3
"""
Build a fixed 50-sample holdout evaluation set from medical data sources.

Strategy:
  - Tag questions by category using keyword matching
  - Stratified sampling across 6 categories
  - Deduplicate against training data (SFT A/B/C)
  - Fixed seed for reproducibility

Usage:
    python tools/experiments/build_eval_set.py \
        --sources data/experiments/sft/medical/ \
        --training_dirs data/experiments/sft/sft_A_medical_only/ \
                        data/experiments/sft/sft_B_medical_general_1_1/ \
                        data/experiments/sft/sft_C_clean_1_1/ \
        --output data/experiments/eval/medical_eval_50.jsonl
"""

import json
import os
import random
import argparse
from collections import defaultdict


# ── Category keyword rules ──
CATEGORY_RULES = {
    "急症": [
        "胸痛", "胸闷", "呼吸困难", "昏迷", "抽搐", "大出血", "咯血",
        "窒息", "休克", "意识丧失", "剧烈腹痛", "持续高烧", "中毒",
        "心脏骤停", "中风", "偏瘫", "言语不清", "一侧面部下垂",
        "突发", "急性", "急诊", "急救", "120", "救护车",
    ],
    "用药": [
        "抗生素", "消炎药", "止痛药", "退烧药", "降压药", "降糖药",
        "儿童用药", "孕妇用药", "哺乳期", "药物过敏", "过量",
        "副作用", "禁忌", "处方药", "OTC", "停药", "换药",
        "剂量", "服用", "口服", "输液", "注射",
    ],
    "常见病": [
        "感冒", "发烧", "咳嗽", "头痛", "胃痛", "腹泻", "便秘",
        "失眠", "过敏", "皮疹", "湿疹", "鼻炎", "咽炎",
        "腰疼", "腿疼", "牙疼", "口腔溃疡", "结膜炎", "流感",
    ],
    "慢病": [
        "高血压", "糖尿病", "冠心病", "哮喘", "慢阻肺", "乙肝",
        "肾炎", "肾病", "甲状腺", "痛风", "关节炎", "骨质疏松",
        "康复", "调理", "饮食控制", "运动建议", "长期服药",
    ],
    "就医": [
        "挂什么科", "看什么医生", "去什么医院", "做什么检查",
        "CT", "MRI", "B超", "血常规", "尿常规", "体检",
        "挂号", "门诊", "住院", "手术", "复查",
    ],
    "通用": [],  # Fallback: doesn't match any medical category
}


def category_weight(category: str) -> int:
    weights = {"急症": 10, "用药": 10, "常见病": 10, "慢病": 8, "就医": 7, "通用": 5}
    return weights.get(category, 0)


def classify_question(question: str) -> str | None:
    """Classify a question into a medical category, or None for purely general."""
    scores = {}
    for cat, keywords in CATEGORY_RULES.items():
        if cat == "通用":
            continue
        score = sum(1 for kw in keywords if kw in question)
        if score > 0:
            scores[cat] = score
    if not scores:
        return "通用"
    return max(scores, key=scores.get)


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


def load_dir(dirpath: str) -> list[dict]:
    rows = []
    if not os.path.isdir(dirpath):
        return rows
    for f in sorted(os.listdir(dirpath)):
        if f.endswith(".jsonl"):
            rows.extend(load_jsonl(os.path.join(dirpath, f)))
    return rows


def extract_questions(rows: list[dict]) -> list[str]:
    qs = []
    for r in rows:
        for turn in r.get("conversations", []):
            if turn.get("from") == "human":
                qs.append(turn.get("value", ""))
    return qs


def main():
    parser = argparse.ArgumentParser(description="Build medical eval set")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Directories or files with raw medical data")
    parser.add_argument("--training_dirs", nargs="+", default=[],
                        help="Training data directories for dedup")
    parser.add_argument("--output", type=str,
                        default="data/experiments/eval/medical_eval_50.jsonl",
                        help="Output file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all source questions
    all_rows = []
    for src in args.sources:
        if os.path.isdir(src):
            all_rows.extend(load_dir(src))
        else:
            all_rows.extend(load_jsonl(src))

    print(f"Loaded {len(all_rows)} source samples")

    # Extract training question texts for dedup
    train_qs = set()
    for td in args.training_dirs:
        train_rows = load_jsonl(os.path.join(td, "train.jsonl"))
        for q in extract_questions(train_rows):
            train_qs.add(q.strip())
    print(f"Training questions for dedup: {len(train_qs)}")

    # Build candidate pool: (id, category, question) tuples, exclude those in training
    pool = defaultdict(list)
    seen_qs = set()
    idx = 0
    for row in all_rows:
        for turn in row.get("conversations", []):
            if turn.get("from") != "human":
                continue
            q = turn.get("value", "").strip()
            if not q or len(q) < 10:
                continue
            if q in train_qs or q in seen_qs:
                continue
            seen_qs.add(q)

            cat = classify_question(q)
            if cat is None:
                continue
            pool[cat].append({"question": q, "category": cat})
            break  # One question per conversation

    print(f"\nCandidate pool by category:")
    for cat in ["急症", "用药", "常见病", "慢病", "就医", "通用"]:
        print(f"  {cat}: {len(pool.get(cat, []))} candidates (target: {category_weight(cat)})")

    # Stratified sampling
    eval_set = []
    for cat in ["急症", "用药", "常见病", "慢病", "就医", "通用"]:
        target = category_weight(cat)
        candidates = pool.get(cat, [])
        if len(candidates) <= target:
            sampled = candidates
        else:
            sampled = random.sample(candidates, target)
        for item in sampled:
            eval_set.append({
                "id": f"{cat}_{len(eval_set) + 1:03d}",
                "category": cat,
                "question": item["question"],
            })
        print(f"  Sampled {len(sampled)} from {cat}")

    # Truncate to target size
    if len(eval_set) > args.size:
        random.shuffle(eval_set)
        eval_set = eval_set[:args.size]

    # Re-index
    cat_counters = defaultdict(int)
    for item in eval_set:
        cat = item["category"]
        cat_counters[cat] += 1
        item["id"] = f"{cat}_{cat_counters[cat]:03d}"

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(eval_set)} samples to {args.output}")
    print("Category distribution:")
    for cat in ["急症", "用药", "常见病", "慢病", "就医", "通用"]:
        n = sum(1 for item in eval_set if item["category"] == cat)
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
