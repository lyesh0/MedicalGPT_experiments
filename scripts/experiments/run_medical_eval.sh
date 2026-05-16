#!/bin/bash
#
# Post-training Evaluation: SFT-C vs DPO vs RLOO vs GRPO
#
# Usage:
#   bash scripts/experiments/run_medical_eval.sh [grpo_checkpoint_name]
#   e.g. bash scripts/experiments/run_medical_eval.sh checkpoint-501
#

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate medicalgpt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

QWEN_BASE="/root/autodl-tmp/models/Qwen3.5-2B"
SFTC_MERGED="outputs/experiments/models/sft_C_merged"
DPO_ADAPTER="outputs/experiments/models/dpo_medical"
RLOO_ADAPTER="outputs/experiments/models/rloo_medical"
GRPO_OUTPUT="outputs/experiments/models/grpo_medical_safety"
GRPO_MERGED="outputs/experiments/models/grpo_medical_safety_merged"
DPO_MERGED="outputs/experiments/models/dpo_medical_merged"
RLOO_MERGED="outputs/experiments/models/rloo_medical_merged"
EVAL_FILE="data/experiments/eval/medical_eval_50.jsonl"
PRED_DIR="outputs/experiments/predictions"
SCORE_DIR="outputs/experiments/scores"

GRPO_CKPT="${1:-}"

# Resolve checkpoint path
if [ -z "$GRPO_CKPT" ]; then
    GRPO_CKPT=$(ls -d "$GRPO_OUTPUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$GRPO_CKPT" ]; then
        echo "[ERROR] No GRPO checkpoint found in $GRPO_OUTPUT"
        exit 1
    fi
elif [ ! -d "$GRPO_CKPT" ]; then
    # Try as a name under GRPO_OUTPUT
    if [ -d "$GRPO_OUTPUT/$GRPO_CKPT" ]; then
        GRPO_CKPT="$GRPO_OUTPUT/$GRPO_CKPT"
    else
        echo "[ERROR] Checkpoint not found: $GRPO_CKPT (also tried $GRPO_OUTPUT/$GRPO_CKPT)"
        exit 1
    fi
fi
echo "[INFO] GRPO checkpoint: $GRPO_CKPT"

echo "============================================"
echo "Medical Safety Evaluation Pipeline"
echo "============================================"

# ── Step 1: Merge adapters ──
echo ""
echo "── Step 1: Merging adapters ──"

# DPO: base=Qwen3.5-2B
if [ -f "$DPO_MERGED/config.json" ]; then
    echo "[OK] DPO already merged: $DPO_MERGED"
else
    echo "Merging DPO (base=Qwen3.5-2B)..."
    python tools/experiments/merge_adapter.py \
        --base_model "$QWEN_BASE" \
        --adapter "$DPO_ADAPTER" \
        --output "$DPO_MERGED"
fi

# RLOO: base=SFT-C merged
if [ -f "$RLOO_MERGED/config.json" ]; then
    echo "[OK] RLOO already merged: $RLOO_MERGED"
else
    echo "Merging RLOO (base=SFT-C merged)..."
    python tools/experiments/merge_adapter.py \
        --base_model "$SFTC_MERGED" \
        --adapter "$RLOO_ADAPTER" \
        --output "$RLOO_MERGED"
fi

# GRPO: base=SFT-C merged
echo "Merging GRPO (base=SFT-C merged)..."
python tools/experiments/merge_adapter.py \
    --base_model "$SFTC_MERGED" \
    --adapter "$GRPO_CKPT" \
    --output "$GRPO_MERGED"

echo "[OK] All models merged"

# ── Step 2: Batch inference ──
echo ""
echo "── Step 2: Running batch inference ──"
mkdir -p "$PRED_DIR"

run_inference() {
    local model_name="$1"
    local model_path="$2"
    local output="$3"

    echo ""
    echo "  Inference: $model_name"
    echo "  Model:      $model_path"

    python -u tools/experiments/batch_inference.py \
        --base_model "$model_path" \
        --lora_model "/tmp/nonexistent_lora" \
        --eval_file "$EVAL_FILE" \
        --output "$output" \
        --max_new_tokens 512 \
        --temperature 0.1
}

run_inference "SFT-C" "$SFTC_MERGED" "$PRED_DIR/sft_C_predictions.jsonl"
run_inference "DPO"   "$DPO_MERGED"  "$PRED_DIR/dpo_predictions.jsonl"
run_inference "RLOO"  "$RLOO_MERGED" "$PRED_DIR/rloo_predictions.jsonl"
run_inference "GRPO"  "$GRPO_MERGED" "$PRED_DIR/grpo_predictions.jsonl"

echo ""
echo "[OK] All inference complete"

# ── Step 3: Score predictions ──
echo ""
echo "── Step 3: Scoring predictions ──"
mkdir -p "$SCORE_DIR"

python -u tools/experiments/score_medical_outputs.py \
    --predictions \
        "$PRED_DIR/sft_C_predictions.jsonl" \
        "$PRED_DIR/dpo_predictions.jsonl" \
        "$PRED_DIR/rloo_predictions.jsonl" \
        "$PRED_DIR/grpo_predictions.jsonl" \
    --labels "SFT-C" "DPO" "RLOO" "GRPO" \
    --output_dir "$SCORE_DIR"

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "  Per-sample: $SCORE_DIR/per_sample_scores.jsonl"
echo "  Summary:    $SCORE_DIR/medical_comparison.csv"
echo "  Report:     $SCORE_DIR/medical_comparison.json"
echo "============================================"
