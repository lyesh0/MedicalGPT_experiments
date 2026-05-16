#!/bin/bash
#
# RM Medical Training — 基于 SFT-C adapter + 医疗安全偏好数据训练 Reward Model
#
# Usage:
#   bash scripts/experiments/run_rm_medical.sh
#

set -euo pipefail

# ── Configuration ──
BASE_MODEL="/root/autodl-tmp/models/Qwen3.5-2B"
SFT_ADAPTER="outputs/experiments/models/sft_C"
TRAIN_DATA="data/experiments/preference_medical_safety"
OUTPUT_DIR="outputs/experiments/models/rm_medical"

# LoRA
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training
LEARNING_RATE="2e-5"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_SOURCE_LENGTH=768
MAX_TARGET_LENGTH=256
WARMUP_STEPS=5
WEIGHT_DECAY=0.001
SAVE_STEPS=500
EVAL_STEPS=50
LOGGING_STEPS=10

# ── Check data ──
if [ ! -f "$TRAIN_DATA/train.jsonl" ]; then
    echo "[ERROR] Preference data not found: $TRAIN_DATA/train.jsonl"
    exit 1
fi

if [ ! -d "$SFT_ADAPTER" ]; then
    echo "[ERROR] SFT adapter not found: $SFT_ADAPTER"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

DATA_COUNT=$(wc -l < "$TRAIN_DATA/train.jsonl")

echo "============================================"
echo "RM Medical Safety Training"
echo "============================================"
echo "  Base model:      $BASE_MODEL"
echo "  SFT adapter:     $SFT_ADAPTER"
echo "  Train data:      $TRAIN_DATA ($DATA_COUNT pairs)"
echo "  Output:          $OUTPUT_DIR"
echo "  LoRA rank:       $LORA_RANK"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Epochs:          $NUM_EPOCHS"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 python training/reward_modeling.py \
    --model_name_or_path "$BASE_MODEL" \
    --peft_path "$SFT_ADAPTER" \
    --train_file_dir "$TRAIN_DATA" \
    --validation_file_dir "$TRAIN_DATA" \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --do_train \
    --use_peft True \
    --seed 42 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --logging_strategy steps \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --bf16 True \
    --fp16 False \
    --torch_dtype bfloat16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --preprocessing_num_workers 4

echo ""
echo "============================================"
echo "RM training complete."
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next: evaluate RM with score_reward_model.py"
echo "============================================"
