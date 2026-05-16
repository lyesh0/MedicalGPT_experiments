#!/usr/bin/env python3
"""
Build medical safety preference dataset via dual-prompt API generation.

Strategy:
  - Load medical questions from a held-out source (not used in SFT training)
  - For each question, call API twice:
      chosen:   safety-oriented system prompt (professional medical assistant)
      rejected: neutral system prompt (direct answer, no safety constraints)
  - Filter pairs where the preference signal is weak or invalid
  - Save in DPO training format (conversations + chosen + rejected)

Usage:
    # Put your key in .env first:
    #   DASHSCOPE_API_KEY=sk-xxx
    #   DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

    python tools/experiments/build_medical_preference_data.py \
        --source data/sft/medical_sft_1K_format.jsonl \
        --output data/experiments/preference_medical_safety/train.jsonl \
        --target 250 \
        --model qwen-plus \
        --eval_file data/experiments/eval/medical_eval_50.jsonl
"""

import json
import os
import random
import time
import argparse
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)


# ── Safety-oriented system prompt (chosen) ──
SYSTEM_SAFE = """你是一个专业的医疗健康助手。在回答任何医疗问题时，你必须遵守以下原则：

1. **安全第一**：对于急症、高危症状（胸痛、呼吸困难、大出血、昏迷、中风等），必须强调立即就医或拨打急救电话，不可给出可能延误就医的建议。
2. **不替代医生**：你的回答不能替代专业医生的诊断。请在回答中适时提醒用户咨询医生。
3. **不提供具体用药方案**：可以解释药物作用和注意事项，但不能给出具体剂量或建议自行用药。
4. **信息准确**：基于循证医学知识回答，不编造医学信息。
5. **回答充分**：完整回答问题，给出有实质内容的建议，同时保持同理心。

请用中文回答。"""

# ── Neutral system prompt (rejected) ──
SYSTEM_NEUTRAL = """你是一个AI助手。请直接回答用户的问题。用中文回答。"""

# ── Safety keywords for filtering ──
SAFETY_KEYWORDS = [
    "就医", "就诊", "看医生", "去医院", "急诊", "门诊",
    "医生", "医师", "请在医生", "遵医嘱", "咨询医生",
    "建议尽快", "立即就医", "拨打急救", "120",
    "不可擅自", "不可随意", "需在医生指导下",
    "不要自行用药", "不要自行服药",
]

DANGER_KEYWORDS = [
    "忍一忍", "不用去医院", "不用看医生",
    "偏方", "土方", "秘方",
    "自己买药", "自己开药", "自行用药", "自行服药",
    "停药试", "不用吃药",
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


def extract_questions(rows: list[dict]) -> list[str]:
    """Extract the first human question from each conversation."""
    qs = []
    for r in rows:
        for turn in r.get("conversations", []):
            role = turn.get("from", "") if isinstance(turn, dict) else ""
            val = turn.get("value", "") if isinstance(turn, dict) else str(turn)
            if role == "human":
                qs.append(val.strip())
                break
    return qs


def load_eval_questions(eval_file: Optional[str]) -> set:
    """Load eval set questions for dedup."""
    if not eval_file or not os.path.exists(eval_file):
        return set()
    rows = load_jsonl(eval_file)
    return {r.get("question", "").strip() for r in rows}


def count_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


# Thread-local client (OpenAI client is not thread-safe across instances)
_thread_local = threading.local()

def _get_client(api_key: str, api_base: str) -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=api_key, base_url=api_base)
    return _thread_local.client


def generate(api_key: str, api_base: str, model: str, system_prompt: str,
             question: str, temperature: float = 0.3, max_tokens: int = 800,
             max_retries: int = 3) -> Optional[str]:
    """Call API to generate a response (thread-safe)."""
    client = _get_client(api_key, api_base)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)
    return None


def process_one_question(api_key: str, api_base: str, model: str,
                         question: str, temperature: float) -> Optional[dict]:
    """Process a single question: generate chosen + rejected, validate, return pair or None."""
    chosen = generate(api_key, api_base, model, SYSTEM_SAFE, question, temperature)
    rejected = generate(api_key, api_base, model, SYSTEM_NEUTRAL, question, temperature)

    if chosen is None or rejected is None:
        return None

    valid, _ = validate_pair(chosen, rejected)
    if not valid:
        return None

    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": chosen,
        "rejected": rejected,
    }


def validate_pair(chosen: str, rejected: str) -> tuple[bool, str]:
    """Validate a chosen/rejected pair. Returns (valid, reason)."""
    if not chosen or not rejected:
        return False, "empty answer"
    if len(chosen) < 50 or len(rejected) < 50:
        return False, f"too short (chosen={len(chosen)}, rejected={len(rejected)})"
    if chosen == rejected:
        return False, "identical"

    # Check content similarity: token overlap ratio on first 200 chars
    def simple_tokens(text):
        return set(text[:200].replace("\n", " ").split())
    c_toks = simple_tokens(chosen)
    r_toks = simple_tokens(rejected)
    if c_toks and r_toks:
        overlap = len(c_toks & r_toks) / len(c_toks | r_toks)
        if overlap > 0.85:
            return False, f"too similar (overlap={overlap:.2f})"

    # Safety signal: chosen should have >= safety keywords than rejected
    # (>= not > because for some factual Qs both may have 0,
    #  the difference is in tone and disclaimer style, not keywords alone)
    chosen_safety = count_keywords(chosen, SAFETY_KEYWORDS)
    rejected_safety = count_keywords(rejected, SAFETY_KEYWORDS)
    if chosen_safety < rejected_safety:
        return False, f"reversed safety (chosen={chosen_safety}, rejected={rejected_safety})"

    # Danger check: rejected should not contain danger keywords
    if count_keywords(rejected, DANGER_KEYWORDS) > 0:
        return False, "rejected contains danger keywords"

    # Bonus point if there IS a clear safety difference
    has_clear_signal = chosen_safety > rejected_safety

    return True, "strong_signal" if has_clear_signal else "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Build medical safety preference data via dual-prompt API generation"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Source medical question file (jsonl, not used in SFT training)")
    parser.add_argument("--output", type=str,
                        default="data/experiments/preference_medical_safety/train.jsonl",
                        help="Output preference data file")
    parser.add_argument("--target", type=int, default=250,
                        help="Target number of valid preference pairs")
    parser.add_argument("--model", type=str, default="qwen3.6",
                        help="API model name")
    parser.add_argument("--eval_file", type=str,
                        default="data/experiments/eval/medical_eval_50.jsonl",
                        help="Eval set file for dedup (skip questions used in eval)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set DASHSCOPE_API_KEY env var / .env file)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL (or set DASHSCOPE_API_BASE env var / .env file)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent API workers")
    args = parser.parse_args()

    # ── Setup ──
    # Load .env file if present
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = val

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("CODEX_API_KEY")
    api_base = (args.api_base or os.environ.get("DASHSCOPE_API_BASE")
                or os.environ.get("CODEX_API_BASE")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = args.model or os.environ.get("QWEN_MODEL", "qwen-plus")

    if not api_key:
        print("ERROR: Set DASHSCOPE_API_KEY in .env or pass --api_key")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)
    random.seed(args.seed)

    # ── Load questions ──
    print(f"Loading source questions from {args.source}")
    source_rows = load_jsonl(args.source)
    questions = extract_questions(source_rows)
    print(f"  Loaded {len(questions)} questions")

    # Filter: remove eval questions, empty questions, short questions
    eval_qs = load_eval_questions(args.eval_file)
    print(f"  Eval questions for dedup: {len(eval_qs)}")

    candidates = []
    for q in questions:
        q = q.strip()
        if not q or len(q) < 10:
            continue
        if q in eval_qs:
            continue
        candidates.append(q)

    # Sort by risk priorty: high-risk keywords first (more likely to produce clear signal)
    HIGH_RISK_KW = ["胸痛", "胸闷", "呼吸困难", "昏迷", "出血", "中毒", "中风",
                     "抗生素", "止痛药", "退烧药", "孕妇", "儿童用药",
                     "急诊", "急救", "手术", "偏方", "停药"]
    def risk_score(q):
        return sum(1 for kw in HIGH_RISK_KW if kw in q)
    candidates.sort(key=risk_score, reverse=True)
    print(f"  Candidates after dedup + filtering: {len(candidates)}")
    print(f"  High-risk candidates (score>0): {sum(1 for q in candidates if risk_score(q) > 0)}")

    # ── Generate (concurrent workers) ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    valid_pairs = []
    api_calls = 0
    lock = threading.Lock()

    print(f"\nGenerating preference pairs (target: {args.target})")
    print(f"  Model: {model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Workers: {args.workers}")
    print(f"  API base: {api_base}")
    print()

    # Use a pool of workers; stop early when target reached
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all candidates
        futures = {}
        for i, q in enumerate(candidates):
            futures[executor.submit(
                process_one_question, api_key, api_base, model, q, args.temperature
            )] = (i, q)

        for future in as_completed(futures):
            idx, q = futures[future]
            with lock:
                api_calls += 2  # 2 calls per question (chosen + rejected)

            if len(valid_pairs) >= args.target:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            try:
                result = future.result()
            except Exception as e:
                print(f"  [FAIL] {q[:60]}... | {e}")
                continue

            if result is None:
                continue

            with lock:
                valid_pairs.append(result)
                # Save incrementally
                with open(args.output, "w", encoding="utf-8") as f:
                    for p in valid_pairs:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")

                c_safety = count_keywords(result["chosen"], SAFETY_KEYWORDS)
                r_safety = count_keywords(result["rejected"], SAFETY_KEYWORDS)
                print(f"[{len(valid_pairs)}/{args.target}] OK | "
                      f"chosen={len(result['chosen'])}chars safety={c_safety} | "
                      f"rejected={len(result['rejected'])}chars safety={r_safety} | "
                      f"Q: {q[:60]}...")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Generation complete.")
    print(f"  API calls:     {api_calls}")
    print(f"  Candidates processed: {len(valid_pairs)} (candidates consumed: see log)")
    print(f"  Valid pairs:   {len(valid_pairs)}")
    print(f"  Output:        {args.output}")

    # Category stats
    cats = Counter()
    for p in valid_pairs:
        q = p["conversations"][0]["value"]
        # Simple keyword classification (reuse eval set logic)
        from build_eval_set import classify_question
        cats[classify_question(q)] += 1

    print(f"\nCategory distribution:")
    for cat in ["急症", "用药", "常见病", "慢病", "就医", "通用"]:
        print(f"  {cat}: {cats.get(cat, 0)}")

    if len(valid_pairs) < args.target:
        print(f"\nWARNING: Only generated {len(valid_pairs)}/{args.target} valid pairs.")
        print("Consider: adding more source questions, or lowering filter strictness.")
        sys.exit(1)

    print(f"\nDone. Next: scripts/experiments/run_dpo_medical.sh")


if __name__ == "__main__":
    main()
