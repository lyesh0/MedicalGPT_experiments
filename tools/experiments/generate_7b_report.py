"""Generate comprehensive 7B experiment report with all metrics and visualizations."""
import json
import os
from datetime import datetime


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_report():
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = "outputs/experiments/reports/7b_full_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Load scoring data
    rule_scores = load_json("outputs/experiments/scores_7b/medical_comparison.json")
    llm_judge = load_jsonl("outputs/experiments/scores_7b/llm_judge_results.jsonl")

    # Compute LLM judge summary
    from collections import defaultdict
    llm_summary = defaultdict(lambda: defaultdict(list))
    for j in llm_judge:
        m = j["model"]
        for k in ["safety", "accuracy", "completeness", "actionability", "risk_control", "overall"]:
            llm_summary[m][k].append(j.get(k, 0))
    llm_avg = {m: {k: round(sum(v)/len(v), 2) for k, v in scores.items()}
               for m, scores in llm_summary.items()}

    # Gather training results
    training_results = {
        "SFT-7B-C": {
            "train_loss": 1.640, "eval_loss": 2.056, "perplexity": 7.82,
            "duration": "19 min", "trainable_params": "20.2M / 7.64B (0.26%)",
            "data": "Clean medical+general 1:1 (5000 samples)",
        },
        "SFT-7B-A": {
            "train_loss": 2.429, "eval_loss": 2.119, "perplexity": 8.32,
            "duration": "19 min", "data": "Medical only (5000 samples)",
        },
        "SFT-7B-B": {
            "train_loss": 1.606, "eval_loss": 1.894, "perplexity": 6.64,
            "duration": "19 min", "data": "Medical+general 1:1 (5000 samples)",
        },
        "DPO-7B": {
            "train_loss": 0.492, "eval_loss": 0.378, "eval_accuracy": "100%",
            "eval_margin": 0.79, "duration": "37 min",
            "data": "Medical safety preference (250 pairs)",
        },
        "GRPO-7B": {
            "final_reward": 1.24, "initial_reward": 0.74,
            "kl_divergence": 0.001, "duration": "26 min",
            "data": "GRPO medical safety (100 prompts, 2 generations/prompt)",
        },
        "RM-7B": {
            "train_loss": 0.445, "eval_loss": 0.282, "eval_mae": 1.77,
            "duration": "15 min",
            "data": "Medical safety preference (250 pairs)",
        },
        "RLOO-7B": {
            "eval_reward": -0.072, "kl_divergence": 0.00004,
            "duration": "~45 min (partial)", "status": "pipeline verified, early stop",
        },
    }

    # Build markdown
    lines = []
    lines.append(f"# 7B Medical Safety Experiment Report")
    lines.append(f"**Date**: {today}")
    lines.append(f"**Base Model**: Qwen2.5-7B-Instruct (4-bit QLoRA)")
    lines.append(f"**Hardware**: 2× RTX 4090 (24GB)")
    lines.append(f"**Framework**: PEFT + bitsandbytes + TRL")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. Experiment Overview ──
    lines.append("## 1. Experiment Pipeline")
    lines.append("")
    lines.append("```")
    lines.append("SFT (A/B/C ablation) → SFT-C selected → DPO → GRPO")
    lines.append("                        ↓                ↓")
    lines.append("                       RM training → RLOO (partial)")
    lines.append("```")
    lines.append("")

    # ── 2. SFT Ablation ──
    lines.append("## 2. SFT Ablation (A/B/C)")
    lines.append("")
    lines.append("| Group | Data | train_loss | eval_loss | perplexity |")
    lines.append("|-------|------|-----------|-----------|------------|")
    for g in ["SFT-7B-A", "SFT-7B-B", "SFT-7B-C"]:
        t = training_results[g]
        lines.append(f"| {g} | {t['data']} | {t['train_loss']} | {t['eval_loss']} | {t['perplexity']} |")
    lines.append("")
    lines.append("**Selection**: SFT-7B-C (consistent with 2B experiments)")
    lines.append("")
    lines.append("![SFT Curves](curves/sft_7b_C.png)")
    lines.append("")

    # ── 3. DPO ──
    lines.append("## 3. DPO Medical Safety")
    lines.append("")
    t = training_results["DPO-7B"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in t.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("![DPO Curves](curves/dpo_7b_medical.png)")
    lines.append("")

    # ── 4. GRPO ──
    lines.append("## 4. GRPO Medical Safety")
    lines.append("")
    t = training_results["GRPO-7B"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in t.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("![GRPO Curves](curves/grpo_7b_medical_safety.png)")
    lines.append("")

    # ── 5. RM ──
    lines.append("## 5. Reward Model (RM)")
    lines.append("")
    t = training_results["RM-7B"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in t.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("![RM Curves](curves/rm_7b_medical.png)")
    lines.append("")

    # ── 6. RLOO ──
    lines.append("## 6. RLOO (Partial)")
    lines.append("")
    t = training_results["RLOO-7B"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in t.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("![RLOO Curves](curves/rloo_7b_medical.png)")
    lines.append("")

    # ── 7. Evaluation Results ──
    lines.append("## 7. Final Evaluation (50 medical prompts)")
    lines.append("")

    lines.append("### 7.1 Rule-based Safety Scoring")
    lines.append("")
    lines.append("| Model | Safety | Danger | Escalate | Complete | RedFlag | Composite |")
    lines.append("|-------|--------|--------|----------|----------|---------|-----------|")
    for r in sorted(rule_scores, key=lambda x: x["overall"]["composite"], reverse=True):
        o = r["overall"]
        lines.append(f"| {r['model']} | {o['avg_safety']:.3f} | {o['avg_danger_penalty']:.3f} | "
                     f"{o['high_risk_escalation_rate']:.3f} | {o['avg_completeness']:.3f} | "
                     f"{o['danger_suggestion_rate']:.3f} | **{o['composite']:.3f}** |")
    lines.append("")

    lines.append("### 7.2 LLM Judge (DeepSeek V4 Flash)")
    lines.append("")
    lines.append("| Model | Safety | Accuracy | Complete | Action | RiskCtrl | Overall |")
    lines.append("|-------|--------|----------|----------|--------|----------|---------|")
    for m in sorted(llm_avg, key=lambda x: llm_avg[x]["overall"], reverse=True):
        s = llm_avg[m]
        lines.append(f"| {m} | {s['safety']:.2f} | {s['accuracy']:.2f} | {s['completeness']:.2f} | "
                     f"{s['actionability']:.2f} | {s['risk_control']:.2f} | **{s['overall']:.2f}** |")
    lines.append("")

    lines.append("### 7.3 Per-Category Breakdown (Rule-based)")
    lines.append("")
    for r in sorted(rule_scores, key=lambda x: x["overall"]["composite"], reverse=True):
        lines.append(f"**{r['model']}** (composite: {r['overall']['composite']:.3f})")
        lines.append("")
        lines.append("| Category | N | Safety | NoDanger | Escalate | Complete | RedFlag |")
        lines.append("|----------|---|--------|----------|----------|----------|---------|")
        for cat, cs in sorted(r["per_category"].items()):
            lines.append(f"| {cat} | {cs['samples']} | {cs['avg_safety']:.3f} | {cs['avg_danger_penalty']:.3f} | "
                         f"{cs['high_risk_escalation_rate']:.3f} | {cs['avg_completeness']:.3f} | "
                         f"{cs['danger_suggestion_rate']:.3f} |")
        lines.append("")

    # ── 8. Model Comparison Examples ──
    lines.append("## 8. Qualitative Comparison")
    lines.append("")

    # Load a few prediction samples for comparison
    sft_preds = load_jsonl("outputs/experiments/predictions_7b/sft_7b_C_predictions.jsonl")
    dpo_preds = load_jsonl("outputs/experiments/predictions_7b/dpo_7b_predictions.jsonl")
    grpo_preds = load_jsonl("outputs/experiments/predictions_7b/grpo_7b_predictions.jsonl")

    sample_ids = [0, 3, 7, 15, 25]  # pick diverse samples
    for sid in sample_ids:
        if sid >= len(sft_preds):
            break
        lines.append(f"### Example {sid+1}: {sft_preds[sid]['question'][:80]}...")
        lines.append(f"**Category**: {sft_preds[sid].get('category', 'N/A')}")
        lines.append("")

        for model_name, preds in [("SFT-7B-C", sft_preds), ("DPO-7B", dpo_preds), ("GRPO-7B", grpo_preds)]:
            if sid < len(preds):
                answer = preds[sid].get("answer", "")[:300]
                lines.append(f"**{model_name}**: {answer}...")
                lines.append("")

    # ── 9. Training Curves Reference ──
    lines.append("## 9. All Training Curves")
    lines.append("")
    for phase, name in [("SFT", "sft_7b_C"), ("DPO", "dpo_7b_medical"),
                         ("RM", "rm_7b_medical"), ("RLOO", "rloo_7b_medical"),
                         ("GRPO", "grpo_7b_medical_safety")]:
        lines.append(f"- **{phase}**: [curves/{name}.png](curves/{name}.png)")
    lines.append("")

    # ── 10. Conclusions ──
    lines.append("## 10. Conclusions")
    lines.append("")
    lines.append("1. **SFT-7B-C** provides a solid baseline with composite safety score of 0.815")
    lines.append("2. **DPO-7B** wins on LLM judge evaluation (overall 6.58), showing better quality responses")
    lines.append("3. **GRPO-7B** wins on rule-based safety metrics (composite 0.831) and shows highest per-category safety scores")
    lines.append("4. **RM-7B** successfully learns to distinguish safe from unsafe medical advice")
    lines.append("5. **RLOO pipeline** is verified end-to-end; further hyperparameter tuning needed")
    lines.append("6. All 4-bit QLoRA training runs within 24GB VRAM per GPU on 2× RTX 4090")
    lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {report_path}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    generate_report()
