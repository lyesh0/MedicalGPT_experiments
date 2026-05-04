#!/bin/bash
#
# SFT Data Pipeline: Download → Analyze → Clean → Build Mixtures
#
# Usage:
#   bash scripts/experiments/run_sft_data_pipeline.sh          # with HF mirror
#   bash scripts/experiments/run_sft_data_pipeline.sh --no-mirror
#

set -euo pipefail

USE_MIRROR="--use_mirror"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-mirror) USE_MIRROR=""; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

MEDICAL_DIR="data/experiments/sft/medical"
GENERAL_DIR="data/experiments/sft/general"
CLEANED_DIR="data/experiments/sft/cleaned"

echo "=========================================="
echo "Step 1/4: Download SFT Datasets"
echo "=========================================="
python tools/experiments/download_sft_data.py $USE_MIRROR --skip_existing

echo ""
echo "=========================================="
echo "Step 2/4: Analyze Raw Data"
echo "=========================================="
mkdir -p "$MEDICAL_DIR" "$GENERAL_DIR"

MEDICAL_JSONL_COUNT=$(find "$MEDICAL_DIR" -name "*.jsonl" 2>/dev/null | wc -l)
GENERAL_JSONL_COUNT=$(find "$GENERAL_DIR" -name "*.jsonl" 2>/dev/null | wc -l)

if [ "$MEDICAL_JSONL_COUNT" -gt 0 ]; then
    python tools/experiments/analyze_sft_data.py \
        --input_dir "$MEDICAL_DIR" \
        --output_stats "$MEDICAL_DIR/analysis.json"
else
    echo "  [WARNING] No medical JSONL files in $MEDICAL_DIR, skipping analysis"
fi

if [ "$GENERAL_JSONL_COUNT" -gt 0 ]; then
    python tools/experiments/analyze_sft_data.py \
        --input_dir "$GENERAL_DIR" \
        --output_stats "$GENERAL_DIR/analysis.json"
else
    echo "  [WARNING] No general JSONL files in $GENERAL_DIR, skipping analysis"
fi

echo ""
echo "=========================================="
echo "Step 3/4: Clean Medical Data"
echo "=========================================="
mkdir -p "$CLEANED_DIR"

if [ "$MEDICAL_JSONL_COUNT" -gt 0 ]; then
    python tools/experiments/clean_sft_data.py \
        --input_dir "$MEDICAL_DIR" \
        --output "$CLEANED_DIR/medical_clean.jsonl" \
        --max_samples 5000
else
    echo "  [WARNING] No medical data to clean, creating empty placeholder"
    touch "$CLEANED_DIR/medical_clean.jsonl"
fi

echo ""
echo "=========================================="
echo "Step 4/4: Build SFT A/B/C Mixtures"
echo "=========================================="

# Determine available sample counts
if [ -f "$CLEANED_DIR/medical_clean.jsonl" ]; then
    CLEAN_COUNT=$(wc -l < "$CLEANED_DIR/medical_clean.jsonl")
else
    CLEAN_COUNT=0
fi

if [ "$MEDICAL_JSONL_COUNT" -gt 0 ]; then
    RAW_COUNT=$(find "$MEDICAL_DIR" -name "*.jsonl" -exec cat {} + | wc -l)
else
    RAW_COUNT=0
fi

if [ "$GENERAL_JSONL_COUNT" -gt 0 ]; then
    GEN_COUNT=$(find "$GENERAL_DIR" -name "*.jsonl" -exec cat {} + | wc -l)
else
    GEN_COUNT=0
fi

echo "  Medical raw: $RAW_COUNT | Medical clean: $CLEAN_COUNT | General: $GEN_COUNT"

# Calculate feasible total: 2 * min(medical_available, general_available)
HALF_TARGET=$(( RAW_COUNT < GEN_COUNT ? RAW_COUNT : GEN_COUNT ))
HALF_TARGET=$(( HALF_TARGET < CLEAN_COUNT ? HALF_TARGET : CLEAN_COUNT ))
# But minimum half is 500 (1000 total)
if [ "$HALF_TARGET" -gt 500 ]; then
    # cap at 2500 per half
    if [ "$HALF_TARGET" -gt 2500 ]; then
        HALF_TARGET=2500
    fi
    TOTAL=$(( HALF_TARGET * 2 ))
else
    TOTAL=0
fi

if [ "$TOTAL" -lt 1000 ]; then
    echo ""
    echo "  [ERROR] Not enough data to build mixtures."
    echo "  Medical raw=$RAW_COUNT, Medical clean=$CLEAN_COUNT, General=$GEN_COUNT"
    echo "  Need at least 1000 total samples (500 per half)."
    echo "  The download step may have failed for some datasets."
    exit 1
fi

echo "  Adjusted total per group: $TOTAL"

python tools/experiments/build_sft_mixture.py \
    --medical_raw "$MEDICAL_DIR" \
    --medical_clean "$CLEANED_DIR/medical_clean.jsonl" \
    --general "$GENERAL_DIR" \
    --total_samples "$TOTAL" \
    --validate

echo ""
echo "=========================================="
echo "Data pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Raw medical:     $MEDICAL_DIR/"
echo "  Raw general:     $GENERAL_DIR/"
echo "  Cleaned medical: $CLEANED_DIR/medical_clean.jsonl"
echo "  SFT-A train:     data/experiments/sft/sft_A_medical_only/train.jsonl"
echo "  SFT-B train:     data/experiments/sft/sft_B_medical_general_1_1/train.jsonl"
echo "  SFT-C train:     data/experiments/sft/sft_C_clean_1_1/train.jsonl"
