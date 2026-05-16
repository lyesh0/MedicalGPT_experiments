#!/usr/bin/env python3
"""
Download SFT datasets via direct HTTP requests to HF mirror.
Bypasses huggingface_hub to avoid network issues on cloud instances.

Usage:
    python tools/experiments/download_sft_data.py --use_mirror
"""

import os
import json
import argparse
import requests

HF_MIRROR = "https://hf-mirror.com"

DATASETS = [
    {
        "repo": "shibing624/medical",
        "files": [
            "finetune/train_zh_0.json",
            "finetune/train_en_1.json",
        ],
        "max_samples": 5000,
        "category": "medical",
        "description": "MedicalGPT medical QA (finetune)",
    },
    {
        "repo": "shibing624/sharegpt_gpt4",
        "files": [
            "sharegpt_gpt4.jsonl",
            "sharegpt_zh_38K_format.jsonl",
        ],
        "max_samples": 10000,
        "category": "general",
        "description": "ShareGPT GPT-4 conversations (EN + ZH)",
    },
]


def normalize_role(role: str) -> str:
    mapping = {"user": "human", "human": "human", "assistant": "gpt",
               "gpt": "gpt", "system": "system"}
    return mapping.get(role.lower().strip(), role.lower().strip())


def row_to_conversations(row: dict) -> list | None:
    if "conversations" in row:
        raw = row["conversations"]
        if isinstance(raw, list) and len(raw) > 0:
            convs = []
            for turn in raw:
                if isinstance(turn, dict):
                    r = normalize_role(turn.get("from", turn.get("role", "")))
                    v = turn.get("value", turn.get("content", ""))
                    if r and str(v).strip():
                        convs.append({"from": r, "value": str(v)})
                elif isinstance(turn, str):
                    convs.append({"from": "human", "value": turn})
            if len(convs) >= 2:
                return convs

    for qk, ak in [("instruction", "output"), ("question", "answer"),
                    ("input", "output"), ("prompt", "response"),
                    ("query", "response")]:
        if qk in row and ak in row:
            human_text = str(row[qk])
            # Merge optional input/context field with instruction
            if qk == "instruction" and "input" in row and str(row["input"]).strip():
                human_text = human_text + "\n" + str(row["input"])
            return [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": str(row[ak])},
            ]

    if "messages" in row and isinstance(row["messages"], list):
        convs = []
        for m in row["messages"]:
            if isinstance(m, dict):
                r = normalize_role(m.get("role", m.get("from", "")))
                c = m.get("content", m.get("value", ""))
                if r and str(c).strip():
                    convs.append({"from": r, "value": str(c)})
        if len(convs) >= 2:
            return convs

    return None


def load_json_file(url: str) -> list[dict]:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return []
    # Try JSONL first (line-delimited JSON objects)
    lines = text.split("\n")
    first = lines[0].strip()
    if first.startswith("{") and first.endswith("}"):
        rows = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    # Fall back to single JSON value (array or object)
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def load_parquet_url(url: str) -> list[dict]:
    import io
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    try:
        import pyarrow.parquet as pq
        return pq.read_table(io.BytesIO(resp.content)).to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(io.BytesIO(resp.content)).to_dict("records")


def download_dataset(ds: dict, max_samples: int, use_mirror: bool) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Downloading: {ds['repo']}")
    print(f"  {ds['description']}")

    base = HF_MIRROR if use_mirror else "https://huggingface.co"
    limit = min(max_samples, ds["max_samples"]) if max_samples else ds["max_samples"]
    fmt = ds.get("format", "json")

    all_rows = []
    for file_path in ds["files"]:
        if len(all_rows) >= limit:
            break
        url = f"{base}/datasets/{ds['repo']}/resolve/main/{file_path}"
        print(f"  Fetching: {url[:80]}...")
        try:
            if fmt == "parquet":
                rows = load_parquet_url(url)
            else:
                rows = load_json_file(url)
            print(f"    Got {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"    [WARN] Failed: {e}")
            continue

    print(f"  Total downloaded: {len(all_rows)} rows")

    converted = []
    failed = 0
    for row in all_rows:
        if len(converted) >= limit:
            break
        convs = row_to_conversations(row)
        if convs and len(convs) >= 2:
            converted.append({"conversations": convs})
        else:
            failed += 1

    print(f"  Converted: {len(converted)} valid, {failed} skipped")
    return converted


def save_jsonl(rows: list[dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved: {filepath} ({len(rows)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--use_mirror", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data/experiments/sft")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    base = args.output_dir
    all_counts = {}

    for ds in DATASETS:
        out_file = f"{ds['repo'].replace('/', '_')}.jsonl"
        out_path = os.path.join(base, ds["category"], out_file)

        if args.skip_existing and os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                count = sum(1 for _ in f)
            print(f"[SKIP] {ds['repo']} - {count} samples")
            all_counts[ds["repo"]] = count
            continue

        rows = download_dataset(ds, args.max_samples or 0, args.use_mirror)
        if not rows:
            print(f"  [WARNING] No data for {ds['repo']}")
            continue
        save_jsonl(rows, out_path)
        all_counts[ds["repo"]] = len(rows)

    print(f"\n{'='*60}")
    print("Download Summary:")
    for ds_name, count in all_counts.items():
        print(f"  {ds_name}: {count} samples")
    print(f"  Total: {sum(all_counts.values())} samples")


if __name__ == "__main__":
    main()
