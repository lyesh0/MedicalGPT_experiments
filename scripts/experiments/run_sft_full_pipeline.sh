#!/bin/bash
#
# Full SFT ablation pipeline: Data → Training → Eval → Scoring
#
# Usage:
#   bash scripts/experiments/run_sft_full_pipeline.sh           # full pipeline
#   bash scripts/experiments/run_sft_full_pipeline.sh --step N  # start from step N
#   bash scripts/experiments/run_sft_full_pipeline.sh --dry_run # print plan only
#
# Steps:
#   1: Download & prepare data
#   2: Train SFT A/B/C
#   3: Build eval set
#   4: Batch inference
#   5: Score & select best SFT
#

set -euo pipefail

START_STEP=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) START_STEP="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

run_step() {
    local step=$1; local desc=$2; local cmd=$3
    if [ "$step" -lt "$START_STEP" ]; then
        echo "[SKIP] Step $step: $desc"
        return
    fi
    echo ""
    echo "############################################"
    echo "## Step $step: $desc"
    echo "############################################"
    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
    else
        eval "$cmd"
    fi
}

# Step 1
run_step 1 "Download & prepare SFT data" \
    "bash scripts/experiments/run_sft_data_pipeline.sh"

# Step 2
run_step 2 "Train SFT-A, SFT-B, SFT-C" \
    "bash scripts/experiments/run_sft_training.sh"

# Step 3
EVAL_FILE="data/experiments/eval/medical_eval_50.jsonl"
run_step 3 "Build evaluation set" \
    "python tools/experiments/build_eval_set.py \
        --sources data/experiments/sft/medical/ \
        --training_dirs data/experiments/sft/sft_A_medical_only/ \
                        data/experiments/sft/sft_B_medical_general_1_1/ \
                        data/experiments/sft/sft_C_clean_1_1/ \
        --output $EVAL_FILE"

# Step 4
INFERENCE_MODELS=(
    "outputs/experiments/models/sft_A"
    "outputs/experiments/models/sft_B"
    "outputs/experiments/models/sft_C"
)
INFERENCE_ARGS=""
for m in "${INFERENCE_MODELS[@]}"; do
    INFERENCE_ARGS="$INFERENCE_ARGS $m"
done

run_step 4 "Batch inference (3 models)" \
    "python tools/experiments/batch_inference.py \
        --base_model Qwen/Qwen3.5-2B \
        --lora_models $INFERENCE_ARGS \
        --eval_file $EVAL_FILE"

# Step 5
PRED_DIR="outputs/experiments/predictions"
run_step 5 "Score & select best SFT" \
    "python tools/experiments/score_medical_outputs.py \
        --predictions $PRED_DIR/sft_A_predictions.jsonl \
                      $PRED_DIR/sft_B_predictions.jsonl \
                      $PRED_DIR/sft_C_predictions.jsonl \
        --labels SFT-A SFT-B SFT-C"

echo ""
echo "============================================"
echo "SFT ablation pipeline complete!"
echo "============================================"
echo ""
echo "Check results:"
echo "  Scores:     outputs/experiments/scores/sft_comparison.csv"
echo "  Predictions: outputs/experiments/predictions/"
echo "  Models:     outputs/experiments/models/"
echo "  Eval set:   $EVAL_FILE"
