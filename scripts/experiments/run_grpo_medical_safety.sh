#!/bin/bash
#
# GRPO Medical Safety Training v2 — with repetition penalty
#

set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate medicalgpt

BASE_MODEL="outputs/experiments/models/sft_C_merged"
PROMPT_DATA="data/experiments/grpo_medical_safety"
OUTPUT_DIR="outputs/experiments/models/grpo_medical_safety_v2"

# LoRA
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training — v2: anti-hacking tuning
LEARNING_RATE="5e-6"
NUM_EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=2
MAX_COMPLETION_LENGTH=128
NUM_GENERATIONS=4
BETA=0.05
WARMUP_STEPS=10
SAVE_STEPS=25
LOGGING_STEPS=5
EVAL_STEPS=25
SAVE_TOTAL_LIMIT=6
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

# ── Check prerequisites ──
if [ ! -f "$PROMPT_DATA/train.jsonl" ]; then
    echo "[ERROR] GRPO prompt data not found: $PROMPT_DATA/train.jsonl"
    exit 1
fi
PROMPT_COUNT=$(wc -l < "$PROMPT_DATA/train.jsonl")
echo "[OK] GRPO prompt data: $PROMPT_COUNT prompts"

if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "[ERROR] Merged SFT-C not found: $BASE_MODEL"
    exit 1
fi

echo ""
echo "============================================"
echo "GRPO Medical Safety Training V2"
echo "============================================"
echo "  v2 changes:"
echo "    + repetition penalty in reward"
echo "    max_completion_length: 256 → 128"
echo "    beta:                  0.02 → 0.05"
echo "    num_generations:       8 → 4"
echo "    save_steps:            50 → 25"
echo "    save_total_limit:      3 → 6"
echo "============================================"
echo "  Base model:      $BASE_MODEL"
echo "  Prompt data:     $PROMPT_DATA ($PROMPT_COUNT prompts)"
echo "  Output:          $OUTPUT_DIR"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Num generations: $NUM_GENERATIONS"
echo "  Beta (KL):       $BETA"
echo "  Batch size:      $BATCH_SIZE x $GRAD_ACCUM accum"
echo "  Completion len:  $MAX_COMPLETION_LENGTH"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/grpo_training.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_file_dir "$PROMPT_DATA" \
    --preprocessing_num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --reward_type medical_safety \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_steps $WARMUP_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --beta $BETA \
    --use_vllm False \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_strategy steps \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --num_generations $NUM_GENERATIONS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --seed $SEED

echo ""
echo "============================================"
echo "GRPO v2 training complete."
echo "Model saved to: $OUTPUT_DIR"
echo "============================================"
