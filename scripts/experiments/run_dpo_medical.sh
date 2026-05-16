#!/bin/bash
#
# DPO Medical Safety Alignment Training
# Trains a new LoRA on Qwen3.5-2B using medical safety preference data.
#
# Usage:
#   bash scripts/experiments/run_dpo_medical.sh
#

set -euo pipefail

# ── Configuration ──
BASE_MODEL="/root/autodl-tmp/models/Qwen3.5-2B"
TRAIN_DATA="data/experiments/preference_medical_safety"
OUTPUT_DIR="outputs/experiments/models/dpo_medical"

# LoRA (same rank as SFT for fair comparison)
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training (DPO beta=0.1 is the TRL default, not exposed via CLI)
LEARNING_RATE="5e-6"
MAX_STEPS=200
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=512
WARMUP_STEPS=20
WEIGHT_DECAY=0.05
SAVE_STEPS=100
LOGGING_STEPS=5
EVAL_STEPS=50

# ── Check data ──
if [ ! -f "$TRAIN_DATA/train.jsonl" ]; then
    echo "[ERROR] Preference data not found: $TRAIN_DATA/train.jsonl"
    echo "  Run: python tools/experiments/build_medical_preference_data.py"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

DATA_COUNT=$(wc -l < "$TRAIN_DATA/train.jsonl")
echo "============================================"
echo "DPO Medical Safety Training"
echo "============================================"
echo "  Base model:      $BASE_MODEL"
echo "  Preference data: $TRAIN_DATA ($DATA_COUNT pairs)"
echo "  Output:          $OUTPUT_DIR"
echo "  LoRA rank:       $LORA_RANK"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Max steps:       $MAX_STEPS"
echo "============================================"

python training/dpo_training.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file_dir "$TRAIN_DATA" \
    --validation_file_dir "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --use_peft True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules all \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --do_train \
    --do_eval \
    --bf16 True \
    --fp16 False \
    --torch_dtype bfloat16 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --preprocessing_num_workers 4 \
    --cache_dir ./cache \
    --lr_scheduler_type cosine \
    --optim adamw_torch

echo ""
echo "============================================"
echo "DPO training complete."
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next: evaluate with batch_inference.py"
echo "============================================"
