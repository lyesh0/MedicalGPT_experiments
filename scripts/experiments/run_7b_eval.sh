#!/bin/bash
# 7B Medical Safety Evaluation Pipeline
# SFT-C vs DPO vs GRPO — rule-based + LLM judge

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate medicalgpt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

BASE_MODEL="outputs/experiments/models/sft_7b_C_merged"
DPO_ADAPTER="outputs/experiments/models/dpo_7b_medical"
GRPO_ADAPTER="outputs/experiments/models/grpo_7b_medical_safety"
EVAL_FILE="data/experiments/eval/medical_eval_50.jsonl"
PRED_DIR="outputs/experiments/predictions_7b"
SCORE_DIR="outputs/experiments/scores_7b"

mkdir -p "$PRED_DIR" "$SCORE_DIR"

echo "============================================"
echo "7B Medical Evaluation Pipeline"
echo "============================================"

# ── Step 1: Batch Inference ──
echo ""
echo "── Step 1: Batch Inference ──"

run_inference() {
    local model_name="$1"
    local lora_path="$2"
    local output="$3"
    echo "  [$model_name] -> $output"
    python -u tools/experiments/batch_inference.py \
        --base_model "$BASE_MODEL" \
        --lora_model "$lora_path" \
        --eval_file "$EVAL_FILE" \
        --output "$output" \
        --max_new_tokens 512 \
        --temperature 0.1
}

# SFT-C baseline (no adapter)
run_inference "SFT-C" "/tmp/__nonexistent_lora" "$PRED_DIR/sft_7b_C_predictions.jsonl"
# DPO
run_inference "DPO" "$DPO_ADAPTER" "$PRED_DIR/dpo_7b_predictions.jsonl"
# GRPO
run_inference "GRPO" "$GRPO_ADAPTER" "$PRED_DIR/grpo_7b_predictions.jsonl"

echo "[OK] All inference complete"

# ── Step 2: Rule-based Scoring ──
echo ""
echo "── Step 2: Rule-based Scoring ──"
python -u tools/experiments/score_medical_outputs.py \
    --predictions \
        "$PRED_DIR/sft_7b_C_predictions.jsonl" \
        "$PRED_DIR/dpo_7b_predictions.jsonl" \
        "$PRED_DIR/grpo_7b_predictions.jsonl" \
    --labels "SFT-7B-C" "DPO-7B" "GRPO-7B" \
    --output_dir "$SCORE_DIR"

# ── Step 3: LLM Judge (DeepSeek V4 Flash) ──
echo ""
echo "── Step 3: LLM Judge (DeepSeek V4 Flash) ──"
python -u tools/experiments/llm_judge.py \
    --predictions \
        "$PRED_DIR/sft_7b_C_predictions.jsonl" \
        "$PRED_DIR/dpo_7b_predictions.jsonl" \
        "$PRED_DIR/grpo_7b_predictions.jsonl" \
    --labels "SFT-7B-C" "DPO-7B" "GRPO-7B" \
    --output_dir "$SCORE_DIR" \
    --max_samples 50

echo ""
echo "============================================"
echo "7B Evaluation complete!"
echo "  Rule-based: $SCORE_DIR/medical_comparison.json"
echo "  LLM judge:  $SCORE_DIR/llm_judge_results.jsonl"
echo "============================================"
