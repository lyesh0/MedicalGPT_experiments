#!/usr/bin/env python3
"""
Download SFT datasets from HuggingFace and save as unified JSONL format.

Usage:
    python tools/experiments/download_sft_data.py --use_mirror
    python tools/experiments/download_sft_data.py --max_samples 3000

Output:
    data/experiments/sft/medical/  -- 3 medical datasets
    data/experiments/sft/general/  -- 1 general dataset
"""

import os
import json
import argparse
import sys

# HF mirror for users in China
HF_MIRROR = "https://hf-mirror.com"

DATASETS = {
    "shibing624/medical": {
        "max_samples": 3000,
        "category": "medical",
        "description": "MedicalGPT author's medical QA dataset",
    },
    "FreedomIntelligence/HuatuoGPT-sft-data-v1": {
        "max_samples": 3000,
        "category": "medical",
        "description": "HuatuoGPT Chinese medical dialogue",
    },
    "wangrongsheng/Medical-Dialogue": {
        "max_samples": 3000,
        "category": "medical",
        "description": "Doctor-patient multi-turn dialogues",
    },
    "shibing624/sharegpt_gpt4": {
        "max_samples": 5000,
        "category": "general",
        "description": "ShareGPT GPT-4 multi-turn conversations",
    },
}

OAI_CONVERSATION_FORMAT = {
    "shibing624/medical": "conversations",
    "FreedomIntelligence/HuatuoGPT-sft-data-v1": "auto",
    "wangrongsheng/Medical-Dialogue": "auto",
    "shibing624/sharegpt_gpt4": "conversations",
}


def normalize_role(role: str) -> str:
    """Unify role names to 'human' / 'gpt' / 'assistant' / 'system'."""
    role_lower = role.lower().strip()
    mapping = {
        "user": "human",
        "human": "human",
        "assistant": "gpt",
        "gpt": "gpt",
        "system": "system",
    }
    return mapping.get(role_lower, role_lower)


def convert_row(row: dict, ds_name: str) -> dict | None:
    """Convert a single row from any source dataset into OAI conversations format."""
    conversations = None

    if "conversations" in row:
        raw = row["conversations"]
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict) and "from" in raw[0] and "value" in raw[0]:
                conversations = raw

    if conversations is None:
        for qk, ak in [
            ("question", "answer"),
            ("input", "output"),
            ("instruction", "output"),
            ("prompt", "response"),
            ("query", "response"),
        ]:
            if qk in row and ak in row:
                conversations = [
                    {"from": "human", "value": str(row[qk])},
                    {"from": "gpt", "value": str(row[ak])},
                ]
                break

    if conversations is None and "text" in row:
        conversations = [
            {"from": "human", "value": str(row["text"])},
            {"from": "gpt", "value": ""},
        ]

    if conversations is None:
        return None

    normalized = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = normalize_role(turn.get("from", turn.get("role", "")))
        value = turn.get("value", turn.get("content", ""))
        if not role or not value:
            continue
        normalized.append({"from": role, "value": str(value)})

    return {"conversations": normalized} if normalized else None


def download_dataset(ds_name: str, config: dict, max_samples: int, use_mirror: bool) -> list[dict]:
    """Download a dataset from HuggingFace and convert to unified format."""
    print(f"\n{'='*60}")
    print(f"Downloading: {ds_name}")
    print(f"  {config['description']}")

    if use_mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print(f"  Using mirror: {HF_MIRROR}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] 'datasets' library not installed. Run: pip install datasets")
        return []

    limit = min(max_samples, config["max_samples"]) if max_samples else config["max_samples"]

    try:
        try:
            ds = load_dataset(ds_name, split="train", trust_remote_code=True)
        except TypeError:
            ds = load_dataset(ds_name, split="train")
        print(f"  Downloaded {len(ds)} total rows")

        rows = []
        failed = 0
        for i, row in enumerate(ds):
            if len(rows) >= limit:
                break
            conv = convert_row(row, ds_name)
            if conv and len(conv["conversations"]) >= 2:
                rows.append(conv)
            else:
                failed += 1

        print(f"  Converted: {len(rows)} valid samples, {failed} skipped")
        return rows

    except Exception as e:
        print(f"  [ERROR] Failed to download {ds_name}: {e}")
        return []


def save_jsonl(rows: list[dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved: {filepath} ({len(rows)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Download SFT datasets for MedicalGPT experiments")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per dataset (0=use built-in limit)")
    parser.add_argument("--use_mirror", action="store_true", help="Use hf-mirror.com for China access")
    parser.add_argument("--output_dir", type=str, default="data/experiments/sft", help="Output root directory")
    parser.add_argument("--skip_existing", action="store_true", help="Skip download if output file exists")
    args = parser.parse_args()

    base = args.output_dir

    all_counts = {}
    for ds_name, config in DATASETS.items():
        out_file = f"{ds_name.replace('/', '_')}.jsonl"
        out_path = os.path.join(base, config["category"], out_file)

        if args.skip_existing and os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                count = sum(1 for _ in f)
            print(f"[SKIP] {ds_name} - already exists ({count} samples)")
            all_counts[ds_name] = count
            continue

        rows = download_dataset(ds_name, config, args.max_samples or 0, args.use_mirror)

        if not rows:
            print(f"  [WARNING] No data downloaded for {ds_name}")
            continue

        save_jsonl(rows, out_path)
        all_counts[ds_name] = len(rows)

    print(f"\n{'='*60}")
    print("Download Summary:")
    for ds, count in all_counts.items():
        print(f"  {ds}: {count} samples")
    print(f"  Total: {sum(all_counts.values())} samples")


if __name__ == "__main__":
    main()
