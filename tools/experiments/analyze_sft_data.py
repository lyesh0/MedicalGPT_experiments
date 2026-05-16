#!/usr/bin/env python3
"""
Analyze SFT data: sample count, length distribution, empty answers,
duplicate rate, role errors, medical keyword coverage.

Usage:
    python tools/experiments/analyze_sft_data.py --input data/experiments/sft/medical/*.jsonl
    python tools/experiments/analyze_sft_data.py --input_dir data/experiments/sft/medical/
"""

import json
import os
import re
import argparse
from collections import Counter
from difflib import SequenceMatcher


MEDICAL_KEYWORDS = [
    "诊断", "治疗", "检查", "手术", "药物", "用药", "剂量",
    "症状", "病因", "感染", "炎症", "慢性", "急性", "住院",
    "出院", "复查", "预防", "疫苗", "手术", "麻醉", "护理",
    "血压", "血糖", "心率", "体温", "疼痛", "出血", "水肿",
    "内科", "外科", "儿科", "妇产科", "急诊", "门诊", "ICU",
    "CT", "MRI", "B超", "X光", "化验", "血液", "尿液",
    "心脏病", "糖尿病", "高血压", "哮喘", "肺炎", "肝炎",
    "抗生素", "消炎药", "止痛药", "降压药", "降糖药",
    "就医", "挂号", "病房", "医嘱", "处方", "医保",
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


def _turn_value(turn) -> str:
    """Extract text from a turn, which may be a dict or a plain string."""
    if isinstance(turn, dict):
        return turn.get("value", turn.get("content", ""))
    return str(turn)


def _turn_role(turn) -> str:
    if isinstance(turn, dict):
        return turn.get("from", turn.get("role", ""))
    return ""


def extract_all_text(conv) -> str:
    if isinstance(conv, list):
        return "".join(_turn_value(t) for t in conv)
    return str(conv)


def extract_human_text(conv) -> str:
    if isinstance(conv, list):
        return "".join(_turn_value(t) for t in conv if _turn_role(t) == "human")
    return ""


def extract_gpt_text(conv) -> str:
    if isinstance(conv, list):
        return "".join(_turn_value(t) for t in conv if _turn_role(t) in ("gpt", "assistant"))
    return ""


def has_empty_answer(row: dict) -> bool:
    for turn in row.get("conversations", []):
        if _turn_role(turn) in ("gpt", "assistant"):
            if len(_turn_value(turn).strip()) < 10:
                return True
    return False


def is_role_valid(row: dict) -> bool:
    valid_roles = {"human", "gpt", "assistant", "system"}
    for turn in row.get("conversations", []):
        if _turn_role(turn) not in valid_roles:
            return False
    return True


def find_duplicates(rows: list[dict], threshold: float = 0.9) -> list[tuple[int, int]]:
    """Count exact + near-duplicate pairs. Uses hashing for exact, sampling for near."""
    texts = [extract_human_text(r.get("conversations", [])).strip() for r in rows]

    # Exact duplicates via hash (O(n))
    from collections import Counter
    text_counts = Counter(texts)
    exact_dup_pairs = sum(c * (c - 1) // 2 for c in text_counts.values() if c > 1)

    # Near-duplicates: only compare distinct non-empty texts, cap at 2000
    distinct = list(set(t for t in texts if t))
    if len(distinct) > 2000:
        import random
        random.seed(42)
        distinct = random.sample(distinct, 2000)

    near_pairs = 0
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if SequenceMatcher(None, distinct[i], distinct[j]).ratio() > threshold:
                near_pairs += 1

    return list(range(exact_dup_pairs + near_pairs))  # length = count, used as len() below


def count_medical_keywords(text: str) -> int:
    return sum(1 for kw in MEDICAL_KEYWORDS if kw in text)


def analyze_file(filepath: str) -> dict:
    rows = load_jsonl(filepath)
    if not rows:
        return {"file": filepath, "total_samples": 0, "error": "No valid JSONL rows"}

    n = len(rows)
    convs = [r.get("conversations", []) for r in rows]

    # Conversation turns
    turn_counts = [len(c) for c in convs]

    # Length stats
    all_text_lens = [len(extract_all_text(c)) for c in convs]
    human_lens = [len(extract_human_text(c)) for c in convs]
    gpt_lens = [len(extract_gpt_text(c)) for c in convs]

    # Empty / short
    empty_answers = sum(1 for r in rows if has_empty_answer(r))
    short_human = sum(1 for l in human_lens if l < 5)

    # Role errors
    role_errors = sum(1 for r in rows if not is_role_valid(r))

    # Duplicates
    dup_pairs = find_duplicates(rows)

    # Medical keyword coverage
    human_keyword_counts = [count_medical_keywords(extract_human_text(c)) for c in convs]
    gpt_keyword_counts = [count_medical_keywords(extract_gpt_text(c)) for c in convs]
    medical_ratio = sum(1 for c in human_keyword_counts if c > 0) / n if n > 0 else 0

    # Language detection
    total_text = "".join(extract_human_text(c) for c in convs)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', total_text))
    total_chars = len(total_text.replace(" ", "").replace("\n", ""))
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

    return {
        "file": filepath,
        "total_samples": n,
        "avg_turns": sum(turn_counts) / n,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "avg_all_len": sum(all_text_lens) / n,
        "avg_human_len": sum(human_lens) / n,
        "avg_gpt_len": sum(gpt_lens) / n,
        "empty_answer_count": empty_answers,
        "empty_answer_ratio": empty_answers / n,
        "short_human_count": short_human,
        "role_error_count": role_errors,
        "duplicate_pairs": len(dup_pairs),
        "duplicate_ratio": len(dup_pairs) / n if n > 0 else 0,
        "medical_keyword_coverage": round(medical_ratio, 3),
        "avg_medical_kw_human": sum(human_keyword_counts) / n if n > 0 else 0,
        "avg_medical_kw_gpt": sum(gpt_keyword_counts) / n if n > 0 else 0,
        "chinese_ratio": round(chinese_ratio, 3),
    }


def print_report(results: list[dict]):
    print(f"\n{'='*80}")
    print("SFT Data Analysis Report")
    print(f"{'='*80}")

    headers = [
        ("File", 35),
        ("Samples", 8),
        ("AvgLen", 7),
        ("Empty%", 7),
        ("Dup%", 6),
        ("RoleErr", 7),
        ("MedCov%", 8),
        ("ZH%", 6),
    ]
    header_line = "".join(h.rjust(w) for h, w in headers)
    print(header_line)
    print("-" * len(header_line))

    total_samples = 0
    for r in results:
        if "error" in r:
            continue
        fname = os.path.basename(r["file"])[:33]
        vals = [
            (fname, 35),
            (str(r["total_samples"]), 8),
            (f"{r['avg_all_len']:.0f}", 7),
            (f"{r['empty_answer_ratio']:.1%}", 7),
            (f"{r['duplicate_ratio']:.1%}", 6),
            (str(r["role_error_count"]), 7),
            (f"{r['medical_keyword_coverage']:.0%}", 8),
            (f"{r['chinese_ratio']:.0%}", 6),
        ]
        print("".join(v.rjust(w) for v, w in vals))
        total_samples += r["total_samples"]

    print("-" * len(header_line))
    print(f"  Total: {total_samples} samples across {len(results)} files")

    print(f"\n{'='*80}")
    print("Data Quality Flags:")
    for r in results:
        if "error" in r:
            continue
        fname = os.path.basename(r["file"])
        flags = []
        if r["empty_answer_ratio"] > 0.1:
            flags.append(f"high empty rate ({r['empty_answer_ratio']:.0%})")
        if r["duplicate_ratio"] > 0.05:
            flags.append(f"many duplicates ({r['duplicate_ratio']:.0%})")
        if r["role_error_count"] > 0:
            flags.append(f"role errors ({r['role_error_count']})")
        if r["chinese_ratio"] < 0.8:
            flags.append(f"low Chinese ratio ({r['chinese_ratio']:.0%})")
        if r["medical_keyword_coverage"] < 0.5:
            flags.append(f"low medical coverage ({r['medical_keyword_coverage']:.0%})")
        status = " [WARNING] " + ", ".join(flags) if flags else " [OK]"
        print(f"  {fname}:{status}")

    print(f"\n{'='*80}")
    print("Recommendation:")
    if total_samples < 3000:
        print("  Samples insufficient for experiment. Consider lowering max_samples or adding more sources.")
    else:
        print(f"  {total_samples} samples available. Proceed with clean_sft_data.py.")


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT data quality")
    parser.add_argument("--input", nargs="+", help="Input JSONL files")
    parser.add_argument("--input_dir", type=str, help="Input directory containing JSONL files")
    parser.add_argument("--output_stats", type=str, help="Save stats as JSON")
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
        print("No input files specified. Use --input or --input_dir.")
        return

    results = [analyze_file(f) for f in sorted(files)]
    print_report(results)

    if args.output_stats:
        with open(args.output_stats, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nStats saved to {args.output_stats}")


if __name__ == "__main__":
    main()
