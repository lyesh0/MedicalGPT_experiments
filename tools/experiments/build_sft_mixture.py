#!/usr/bin/env python3
"""
Build SFT mixture datasets for A/B/C ablation experiments.

    SFT-A: medical only
    SFT-B: medical + general (1:1)
    SFT-C: cleaned medical + general (1:1)

Usage:
    python tools/experiments/build_sft_mixture.py \
        --medical_raw data/experiments/sft/medical/ \
        --medical_clean data/experiments/sft/cleaned/medical_clean.jsonl \
        --general data/experiments/sft/general/ \
        --total_samples 5000 \
        --output_dir data/experiments/sft/

All three groups get the same total number of training samples.
"""

import json
import os
import random
import argparse


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


def save_jsonl(rows: list[dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return rows[:]
    return random.sample(rows, n)


def build_group_a(medical_raw: list[dict], total: int) -> list[dict]:
    """SFT-A: 100% medical raw data."""
    return sample_rows(medical_raw, total)


def build_group_b(medical_raw: list[dict], general: list[dict], total: int) -> list[dict]:
    """SFT-B: 50% medical raw + 50% general, 1:1 ratio."""
    half = total // 2
    med = sample_rows(medical_raw, half)
    gen = sample_rows(general, half)
    combined = med + gen
    random.shuffle(combined)
    return combined


def build_group_c(medical_clean: list[dict], general: list[dict], total: int) -> list[dict]:
    """SFT-C: 50% cleaned medical + 50% general, 1:1 ratio."""
    half = total // 2
    med = sample_rows(medical_clean, half)
    gen = sample_rows(general, half)
    combined = med + gen
    random.shuffle(combined)
    return combined


def validate_row(row: dict) -> list[str]:
    errors = []
    conv = row.get("conversations", [])
    if not conv:
        errors.append("empty conversations")
        return errors
    roles = [turn.get("from") for turn in conv]
    if "human" not in roles:
        errors.append("no human turn")
    if not any(r in ("gpt", "assistant") for r in roles):
        errors.append("no assistant turn")
    for turn in conv:
        if not turn.get("value", "").strip():
            errors.append(f"empty value for {turn.get('from')}")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT mixture datasets for A/B/C ablation"
    )
    parser.add_argument("--medical_raw", type=str, required=True,
                        help="Path to raw medical data directory or jsonl file")
    parser.add_argument("--medical_clean", type=str, required=True,
                        help="Path to cleaned medical data jsonl file")
    parser.add_argument("--general", type=str, required=True,
                        help="Path to general data directory or jsonl file")
    parser.add_argument("--total_samples", type=int, default=5000,
                        help="Total samples per group")
    parser.add_argument("--output_dir", type=str, default="data/experiments/sft",
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Validate output samples")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    if os.path.isdir(args.medical_raw):
        medical_raw = load_dir(args.medical_raw)
    else:
        medical_raw = load_jsonl(args.medical_raw)

    if os.path.isdir(args.general):
        general = load_dir(args.general)
    else:
        general = load_jsonl(args.general)

    medical_clean = load_jsonl(args.medical_clean)

    print(f"Loaded data:")
    print(f"  Medical raw:    {len(medical_raw)} samples")
    print(f"  Medical clean:  {len(medical_clean)} samples")
    print(f"  General:        {len(general)} samples")
    print(f"  Target/group:   {args.total_samples}")

    # Check availability
    if len(medical_raw) < args.total_samples:
        print(f"  [WARNING] Raw medical ({len(medical_raw)}) < target ({args.total_samples})")
    if len(medical_clean) < args.total_samples // 2:
        print(f"  [WARNING] Clean medical ({len(medical_clean)}) < half target ({args.total_samples // 2})")
    if len(general) < args.total_samples // 2:
        print(f"  [WARNING] General ({len(general)}) < half target ({args.total_samples // 2})")

    # Adjust total if needed
    total = args.total_samples
    min_available = min(len(medical_raw), len(medical_clean), len(general))
    if min_available < total // 2:
        total = min_available * 2
        print(f"  Adjusted target to {total} due to data limits")

    # Build three groups
    groups = {
        "sft_A_medical_only": build_group_a(medical_raw, total),
        "sft_B_medical_general_1_1": build_group_b(medical_raw, general, total),
        "sft_C_clean_1_1": build_group_c(medical_clean, general, total),
    }

    # Save
    for dirname, rows in groups.items():
        out_path = os.path.join(args.output_dir, dirname, "train.jsonl")
        save_jsonl(rows, out_path)
        print(f"\n  [{dirname}]")
        print(f"    Saved: {out_path}")
        print(f"    Samples: {len(rows)}")

        if args.validate:
            errors = 0
            for row in rows:
                row_errors = validate_row(row)
                if row_errors:
                    errors += 1
            print(f"    Validation errors: {errors}/{len(rows)}")
            if errors > 0:
                print(f"    [WARNING] {errors} samples have format issues")

    # Summary
    print(f"\n{'='*60}")
    print("Mixture Summary:")
    print(f"  Random seed:   {args.seed}")
    print(f"  Samples/group: {total}")
    print(f"  SFT-A:  medical only")
    print(f"  SFT-B:  medical + general (1:1)")
    print(f"  SFT-C:  clean medical + general (1:1)")

    # Training config reminder
    print(f"\n{'='*60}")
    print("Next: run SFT training for each group")
    for dirname in groups:
        adapter = f"outputs/experiments/models/{dirname}"
        data = os.path.join(args.output_dir, dirname)
        print(f"  # {dirname}")
        print(f"  python training/supervised_finetuning.py \\")
        print(f"      --model_name_or_path Qwen/Qwen3.5-2B \\")
        print(f"      --train_file_dir {data} \\")
        print(f"      --output_dir {adapter} \\")
        print(f"      --use_peft True --lora_rank 8 --lora_alpha 16 \\")
        print(f"      --num_train_epochs 1 --per_device_train_batch_size 2 \\")
        print(f"      --gradient_accumulation_steps 8 \\")
        print()


if __name__ == "__main__":
    main()
