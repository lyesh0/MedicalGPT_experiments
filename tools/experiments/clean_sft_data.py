#!/usr/bin/env python3
"""
Clean SFT data: filter empty answers, short responses, duplicates,
non-Chinese content, and potentially dangerous medical advice.

Usage:
    python tools/experiments/clean_sft_data.py \
        --input data/experiments/sft/medical/ \
        --output data/experiments/sft/cleaned/medical_clean.jsonl

    python tools/experiments/clean_sft_data.py \
        --input_dir data/experiments/sft/medical/ \
        --output_dir data/experiments/sft/cleaned/
"""

import json
import os
import re
import random
import argparse
from difflib import SequenceMatcher


# Medical red flags: patterns that suggest dangerous or irresponsible advice
DANGEROUS_PATTERNS = [
    r"不用去[医看]",
    r"不用[医看].*[医看]",
    r"忍一忍.*好[了啦]",
    r"偏方.*[治医]",
    r"停药.*[观查].*看",
    r"随便吃.*药",
    r"自己[买开].*药",
    r"不用.*处方",
    r"百度.*[查搜]",
    r"网上说",
    r"不用打.*疫苗",
]


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


def get_answer_texts(row: dict) -> list[str]:
    return [
        turn.get("value", "")
        for turn in row.get("conversations", [])
        if turn.get("from") in ("gpt", "assistant")
    ]


def get_question_texts(row: dict) -> list[str]:
    return [
        turn.get("value", "")
        for turn in row.get("conversations", [])
        if turn.get("from") == "human"
    ]


def get_all_text(row: dict) -> str:
    return "".join(turn.get("value", "") for turn in row.get("conversations", []))


def chinese_char_ratio(text: str) -> float:
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    total = len(text.replace(" ", "").replace("\n", ""))
    return chinese / total if total > 0 else 0


def has_dangerous_content(text: str) -> bool:
    return any(re.search(p, text) for p in DANGEROUS_PATTERNS)


def filter_empty_answers(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keep, removed = [], []
    for r in rows:
        answers = get_answer_texts(r)
        if all(len(a.strip()) < 10 for a in answers):
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed


def filter_short_questions(rows: list[dict], min_chars: int = 5) -> tuple[list[dict], list[dict]]:
    keep, removed = [], []
    for r in rows:
        questions = get_question_texts(r)
        if any(len(q.strip()) < min_chars for q in questions):
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed


def filter_low_chinese(rows: list[dict], threshold: float = 0.5) -> tuple[list[dict], list[dict]]:
    keep, removed = [], []
    for r in rows:
        if chinese_char_ratio(get_all_text(r)) >= threshold:
            keep.append(r)
        else:
            removed.append(r)
    return keep, removed


def filter_role_errors(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid_roles = {"human", "gpt", "assistant", "system"}
    keep, removed = [], []
    for r in rows:
        conv = r.get("conversations", [])
        if all(turn.get("from") in valid_roles for turn in conv):
            keep.append(r)
        else:
            removed.append(r)
    return keep, removed


def filter_dangerous(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keep, removed = [], []
    for r in rows:
        if has_dangerous_content(get_all_text(r)):
            removed.append(r)
        else:
            keep.append(r)
    return keep, removed


def deduplicate(rows: list[dict], threshold: float = 0.9) -> list[dict]:
    """Remove near-duplicates based on first question text similarity."""
    if len(rows) <= 1:
        return rows

    questions = [get_question_texts(r)[0] if get_question_texts(r) else "" for r in rows]
    keep_indices = set(range(len(rows)))
    seen_texts = set()

    for i in range(len(questions)):
        if i not in keep_indices:
            continue
        qi = questions[i]
        # Exact match check (fast)
        if qi in seen_texts:
            keep_indices.discard(i)
            continue
        seen_texts.add(qi)

        # Near duplicate check (slow, only against kept items)
        for j in range(i + 1, len(questions)):
            if j not in keep_indices:
                continue
            if SequenceMatcher(None, qi, questions[j]).ratio() > threshold:
                keep_indices.discard(j)

    return [rows[i] for i in sorted(keep_indices)]


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"input": len(rows)}

    rows, removed_empty = filter_empty_answers(rows)
    stats["removed_empty_answer"] = len(removed_empty)

    rows, removed_short = filter_short_questions(rows)
    stats["removed_short_question"] = len(removed_short)

    rows, removed_zh = filter_low_chinese(rows)
    stats["removed_low_chinese"] = len(removed_zh)

    rows, removed_role = filter_role_errors(rows)
    stats["removed_role_error"] = len(removed_role)

    rows, removed_danger = filter_dangerous(rows)
    stats["removed_dangerous"] = len(removed_danger)

    before_dedup = len(rows)
    rows = deduplicate(rows)
    stats["removed_duplicate"] = before_dedup - len(rows)

    random.seed(42)
    random.shuffle(rows)
    stats["output"] = len(rows)

    return rows, stats


def main():
    parser = argparse.ArgumentParser(description="Clean SFT data")
    parser.add_argument("--input", nargs="+", help="Input JSONL files")
    parser.add_argument("--input_dir", type=str, help="Input directory of JSONL files")
    parser.add_argument("--output", type=str, help="Output JSONL file (for --input)")
    parser.add_argument("--output_dir", type=str, help="Output directory (for --input_dir)")
    parser.add_argument("--max_samples", type=int, default=0, help="Max output samples (0=no limit)")
    args = parser.parse_args()

    files = []
    if args.input:
        files = args.input
    if args.input_dir:
        files.extend(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".jsonl")
        )

    if not files:
        print("No input files. Use --input or --input_dir.")
        return

    all_rows = []
    for f in files:
        rows = load_jsonl(f)
        print(f"Loaded {len(rows)} samples from {os.path.basename(f)}")
        all_rows.extend(rows)

    print(f"\nTotal loaded: {len(all_rows)}")

    cleaned, stats = clean(all_rows)

    if args.max_samples and args.max_samples > 0 and len(cleaned) > args.max_samples:
        random.seed(42)
        cleaned = random.sample(cleaned, args.max_samples)
        stats["max_samples_truncated"] = args.max_samples

    # Determine output path
    if args.output:
        out_path = args.output
    elif args.output_dir:
        out_path = os.path.join(args.output_dir, "medical_clean.jsonl")
    else:
        print("Specify --output or --output_dir")
        return

    save_jsonl(cleaned, out_path)

    print(f"\n{'='*50}")
    print("Cleaning Summary:")
    for step, count in stats.items():
        print(f"  {step}: {count}")
    print(f"  Output file: {out_path}")


if __name__ == "__main__":
    main()
