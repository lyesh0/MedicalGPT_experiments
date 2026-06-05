#!/bin/bash
#
# RLOO 7B Medical Safety Training — SFT-C merged policy + trained RM → RLOO
#
# Usage:
#   bash scripts/experiments/run_rloo_medical_7b.sh
#
# Prerequisites:
#   1. Merged SFT-C:       outputs/experiments/models/sft_7b_C_merged
#   2. Trained RM:         outputs/experiments/models/rm_7b_medical
#   3. RM validated:       MAE < 1.0, pairwise accuracy > 70%
#      (run: python tools/experiments/score_reward_model.py ...)
#

set -euo pipefail

# ── Configuration ──
MERGED_SFT="outputs/experiments/models/sft_7b_C_merged"
RM_MODEL="outputs/experiments/models/rm_7b_medical"
OUTPUT_DIR="outputs/experiments/models/rloo_7b_medical"

# RLOO prompt data
PROMPT_DATA_DIR="data/experiments/rloo_prompts_7b"

# LoRA (conservative)
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training (aligned with 2B RLOO settings)
LEARNING_RATE="1e-5"
MAX_STEPS=200
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_SOURCE_LENGTH=768
MAX_COMPLETION_LENGTH=256
NUM_GENERATIONS=8
WARMUP_STEPS=10
WEIGHT_DECAY=0.01
SAVE_STEPS=50
LOGGING_STEPS=5
EVAL_STEPS=50
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

# ── Check merged SFT ──
if [ ! -f "$MERGED_SFT/config.json" ]; then
    echo "[ERROR] Merged SFT-C not found: $MERGED_SFT"
    exit 1
fi

# ── Check RM model ──
if [ ! -d "$RM_MODEL" ]; then
    echo "[ERROR] RM model not found: $RM_MODEL"
    echo "  Run: bash scripts/experiments/run_rm_medical_7b.sh"
    echo "  Then validate RM with score_reward_model.py before running RLOO."
    exit 1
fi

# ── Prepare RLOO prompt data ──
mkdir -p "$PROMPT_DATA_DIR"

if [ ! -f "$PROMPT_DATA_DIR/train.jsonl" ]; then
    echo "============================================"
    echo "Preparing RLOO prompt data"
    echo "============================================"
    python -c "
import json, random
random.seed(42)

# Use medical SFT data as prompt source
with open('data/sft/medical_sft_1K_format.jsonl') as f:
    data = [json.loads(l) for l in f]

random.shuffle(data)
split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]

with open('$PROMPT_DATA_DIR/train.jsonl', 'w') as f:
    for d in train_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

with open('$PROMPT_DATA_DIR/validation.jsonl', 'w') as f:
    for d in val_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'Train prompts: {len(train_data)}, Val prompts: {len(val_data)}')
"
else
    echo "[OK] RLOO prompt data already exists: $PROMPT_DATA_DIR"
fi

echo ""
echo "============================================"
echo "RLOO 7B Medical Safety Training"
echo "============================================"
echo "  Policy (merged SFT-C): $MERGED_SFT"
echo "  Reward model:           $RM_MODEL"
echo "  Prompt data:            $PROMPT_DATA_DIR"
echo "  Output:                 $OUTPUT_DIR"
echo "  LoRA rank:              $LORA_RANK"
echo "  Learning rate:          $LEARNING_RATE"
echo "  Max steps:              $MAX_STEPS"
echo "  Num generations:        $NUM_GENERATIONS"
echo "  Max completion length:  $MAX_COMPLETION_LENGTH"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 python training/ppo_training.py \
    --model_name_or_path "$MERGED_SFT" \
    --sft_model_path "$MERGED_SFT" \
    --reward_model_path "$RM_MODEL" \
    --train_file_dir "$PROMPT_DATA_DIR" \
    --validation_file_dir "$PROMPT_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --use_peft True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --trust_remote_code True \
    --dtype bfloat16 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --generation_batch_size 16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --save_steps $SAVE_STEPS \
    --save_strategy steps \
    --save_total_limit 3 \
    --logging_steps $LOGGING_STEPS \
    --logging_strategy steps \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --do_train True \
    --do_eval True \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing True \
    --seed $SEED \
    --remove_unused_columns False \
    --preprocessing_num_workers 2 \
    --report_to tensorboard \
    --cache_dir ./cache \
    --ddp_timeout 30000

echo ""
echo "============================================"
echo "7B RLOO training complete."
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next: evaluate with batch_inference.py"
echo "============================================"
