"""
Build GRPO prompt data from medical SFT conversations.
Assigns categories via keyword matching for scenario-aware reward.
"""

import json
import argparse
import random
from collections import Counter

CATEGORY_RULES = [
    ("急症", [
        "胸痛", "胸闷", "呼吸困难", "抽搐", "昏迷", "休克", "大出血",
        "咯血", "呕血", "高热", "意识", "窒息", "剧烈", "急性",
        "突然晕倒", "心跳骤停", "中风", "脑出血", "心梗",
    ]),
    ("用药", [
        "吃什么药", "用药", "服药", "药物", "剂量", "停药", "换药",
        "副作用", "抗生素", "消炎药", "降压药", "降糖药", "处方药",
        "吃药", "药品", "药量", "口服", "外用",
    ]),
    ("慢病", [
        "糖尿病", "高血压", "冠心病", "慢性", "肝硬化", "肾病",
        "痛风", "哮喘", "慢阻肺", "乙肝", "甲亢", "类风湿",
        "血糖", "血压", "血脂", "尿酸",
    ]),
    ("就医", [
        "挂什么科", "看什么科", "去哪个医院", "做什么检查", "要不要去",
        "需不需要去", "挂号", "就诊", "看医生", "门诊", "住院",
        "手术", "什么检查", "挂哪个",
    ]),
    ("常见病", [
        "感冒", "发烧", "咳嗽", "头痛", "肚子疼", "腹泻", "便秘",
        "失眠", "过敏", "皮疹", "腰疼", "关节疼", "胃疼",
        "恶心", "呕吐", "头晕", "乏力", "嗓子疼", "流鼻涕",
    ]),
]


def assign_category(question: str) -> str:
    question_lower = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_RULES:
        scores[cat] = sum(1 for kw in keywords if kw in question)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "通用"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (ShareGPT format)")
    parser.add_argument("--output_dir", required=True, help="Output directory for GRPO data")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples (-1 for all)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        raw_data = [json.loads(line) for line in f]

    # Extract user questions from ShareGPT conversations
    data = []
    for item in raw_data:
        conversations = item.get("conversations", [])
        question = ""
        for turn in conversations:
            if turn.get("from") == "human":
                question = turn.get("value", "")
                break
        if question:
            data.append(question)

    random.shuffle(data)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Assign categories
    results = []
    cats = Counter()
    for q in data:
        cat = assign_category(q)
        cats[cat] += 1
        results.append({"question": q, "category": cat})

    # Split
    split = int(len(results) * (1 - args.val_ratio))
    train_data = results[:split]
    val_data = results[split:]

    with open(os.path.join(args.output_dir, "train.jsonl"), "w") as f:
        for d in train_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    val_file = os.path.join(args.output_dir, "validation.jsonl")
    if val_data:
        with open(val_file, "w") as f:
            for d in val_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print("Category distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count} ({100*count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
