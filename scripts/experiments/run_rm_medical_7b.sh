#!/bin/bash
#
# RM Medical Training (7B) — 基于 pretrained 基座 + SFT-C adapter + 医疗安全偏好数据训练 Reward Model
#
# Usage:
#   bash scripts/experiments/run_rm_medical_7b.sh
#
# Prerequisites:
#   - Pretrained model: /root/autodl-tmp/models/Qwen2.5-7B-Instruct
#   - SFT-C adapter:    outputs/experiments/models/sft_7b_C
#   - Preference data:  data/experiments/preference_medical_safety/train.jsonl
#
# Post-check (mandatory before RLOO):
#   python tools/experiments/score_reward_model.py \
#       --model_path outputs/experiments/models/rm_7b_medical \
#       --base_model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
#       --peft_path outputs/experiments/models/sft_7b_C \
#       --data data/experiments/preference_medical_safety/train.jsonl
#

set -euo pipefail

# ── Configuration ──
BASE_MODEL="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
SFT_ADAPTER="outputs/experiments/models/sft_7b_C"
TRAIN_DATA="data/experiments/preference_medical_safety"
OUTPUT_DIR="outputs/experiments/models/rm_7b_medical"
CACHE_DIR="/root/autodl-tmp/cache"

# LoRA
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training (conservative for 7B QLoRA)
LEARNING_RATE="2e-5"
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_SOURCE_LENGTH=768
MAX_TARGET_LENGTH=256
WARMUP_STEPS=5
WEIGHT_DECAY=0.001
SAVE_STEPS=100
EVAL_STEPS=25
LOGGING_STEPS=5
SEED=42
MAX_TRAIN_SAMPLES=-1
MAX_EVAL_SAMPLES=10
PREPROCESS_WORKERS=2

# ── Check pretrained model ──
if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "[ERROR] Pretrained model not found: $BASE_MODEL"
    exit 1
fi

# ── Check SFT adapter ──
if [ ! -d "$SFT_ADAPTER" ]; then
    echo "[ERROR] SFT adapter not found: $SFT_ADAPTER"
    exit 1
fi

# ── Check preference data ──
if [ ! -f "$TRAIN_DATA/train.jsonl" ]; then
    echo "[ERROR] Preference data not found: $TRAIN_DATA/train.jsonl"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

DATA_COUNT=$(wc -l < "$TRAIN_DATA/train.jsonl")
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "RM 7B Medical Safety Training (QLoRA)"
echo "============================================"
echo "  Pretrained base:  $BASE_MODEL"
echo "  SFT adapter:      $SFT_ADAPTER"
echo "  Train data:       $TRAIN_DATA ($DATA_COUNT pairs)"
echo "  Output:           $OUTPUT_DIR"
echo "  LoRA rank:        $LORA_RANK"
echo "  Learning rate:    $LEARNING_RATE"
echo "  Epochs:           $NUM_EPOCHS"
echo "============================================"
echo ""
echo "NOTE: RM trains on pretrained base + SFT adapter (separated),"
echo "      NOT on merged SFT. This prevents representation bias."
echo ""

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
    --load_in_4bit True \
    --seed $SEED \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
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
    --cache_dir "$CACHE_DIR" \
    --preprocessing_num_workers $PREPROCESS_WORKERS

echo ""
echo "============================================"
echo "7B RM training complete."
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "MANDATORY: Evaluate RM before using in RLOO:"
echo "  python tools/experiments/score_reward_model.py \\"
echo "      --model_path $OUTPUT_DIR \\"
echo "      --base_model $BASE_MODEL \\"
echo "      --peft_path $SFT_ADAPTER \\"
echo "      --data $TRAIN_DATA/train.jsonl"
echo ""
echo "Only proceed to RLOO if MAE < 1.0 and pairwise accuracy > 70%."
echo "============================================"
