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

echo "=========================================="
echo "Step 1/4: Download SFT Datasets"
echo "=========================================="
python tools/experiments/download_sft_data.py $USE_MIRROR --skip_existing

echo ""
echo "=========================================="
echo "Step 2/4: Analyze Raw Data"
echo "=========================================="
python tools/experiments/analyze_sft_data.py \
    --input_dir data/experiments/sft/medical/ \
    --output_stats data/experiments/sft/medical/analysis.json

echo ""
echo "=========================================="
echo "Step 3/4: Clean Medical Data"
echo "=========================================="
python tools/experiments/clean_sft_data.py \
    --input_dir data/experiments/sft/medical/ \
    --output data/experiments/sft/cleaned/medical_clean.jsonl \
    --max_samples 5000

echo ""
echo "=========================================="
echo "Step 4/4: Build SFT A/B/C Mixtures"
echo "=========================================="
python tools/experiments/build_sft_mixture.py \
    --medical_raw data/experiments/sft/medical/ \
    --medical_clean data/experiments/sft/cleaned/medical_clean.jsonl \
    --general data/experiments/sft/general/ \
    --total_samples 5000 \
    --validate

echo ""
echo "=========================================="
echo "Data pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Raw medical:     data/experiments/sft/medical/"
echo "  Raw general:     data/experiments/sft/general/"
echo "  Cleaned medical: data/experiments/sft/cleaned/medical_clean.jsonl"
echo "  SFT-A train:     data/experiments/sft/sft_A_medical_only/train.jsonl"
echo "  SFT-B train:     data/experiments/sft/sft_B_medical_general_1_1/train.jsonl"
echo "  SFT-C train:     data/experiments/sft/sft_C_clean_1_1/train.jsonl"
