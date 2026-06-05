#!/bin/bash
#
# Run GRPO training for a 7B model with conservative QLoRA settings.
#
# Usage:
#   bash scripts/experiments/run_grpo_medical_safety_7b.sh
#   bash scripts/experiments/run_grpo_medical_safety_7b.sh --dry_run
#

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-outputs/experiments/models/sft_7b_best_merged}"
TRAIN_DATA="${TRAIN_DATA:-data/experiments/grpo/medical_safety}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/models/grpo_7b_medical_safety}"
DRY_RUN=false

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE=1
GRAD_ACCUM=4
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
SAVE_STEPS=25
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=5
EVAL_STEPS=25
TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"
ACTIVATE_ENV="${ACTIVATE_ENV:-true}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-medicalgpt}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$ACTIVATE_ENV" == "true" ]] && [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "[ERROR] Base model not found: $BASE_MODEL"
    echo "        Merge the best 7B SFT adapter before running GRPO."
    exit 1
fi

if [[ ! -f "$TRAIN_DATA/train.jsonl" ]]; then
    echo "[ERROR] GRPO training data not found: $TRAIN_DATA/train.jsonl"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "============================================"
echo "GRPO 7B Conservative Training"
echo "Base model:        $BASE_MODEL"
echo "Train data:        $TRAIN_DATA"
echo "Output dir:        $OUTPUT_DIR"
echo "Train samples:     $TRAIN_SAMPLES"
echo "Num generations:   $NUM_GENERATIONS"
echo "Completion length: $MAX_COMPLETION_LENGTH"
echo "QLoRA:             enabled"
echo "============================================"

CMD="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/grpo_training.py \
    --model_name_or_path $BASE_MODEL \
    --train_file_dir $TRAIN_DATA \
    --train_samples $TRAIN_SAMPLES \
    --preprocessing_num_workers 2 \
    --output_dir $OUTPUT_DIR \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_steps $MAX_STEPS \
    --beta 0.05 \
    --use_vllm False \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_strategy steps \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --num_generations $NUM_GENERATIONS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --reward_type medical_safety \
    --max_completion_length $MAX_COMPLETION_LENGTH"

if $DRY_RUN; then
    echo "[DRY RUN] Command:"
    echo "$CMD" | sed 's/    /  /g'
else
    eval "$CMD"
fi

echo ""
echo "============================================"
echo "7B GRPO training complete."
echo "============================================"
