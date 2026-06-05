#!/bin/bash
#
# Run DPO training for a 7B model with conservative QLoRA settings.
#
# Usage:
#   bash scripts/experiments/run_dpo_training_7b.sh
#   bash scripts/experiments/run_dpo_training_7b.sh --dry_run
#

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-outputs/experiments/models/sft_7b_best_merged}"
TRAIN_DATA="${TRAIN_DATA:-data/reward}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/models/dpo_7b_medical}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cache}"
DRY_RUN=false

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LEARNING_RATE="5e-6"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-768}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-256}"
WARMUP_STEPS=10
WEIGHT_DECAY=0.05
SAVE_STEPS=25
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=5
EVAL_STEPS=25
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-10}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-2}"
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
    echo "        Merge the best 7B SFT adapter before running DPO."
    exit 1
fi

if [[ ! -d "$TRAIN_DATA" ]]; then
    echo "[ERROR] Training data directory not found: $TRAIN_DATA"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" "$CACHE_DIR"

echo "============================================"
echo "DPO 7B Conservative Training"
echo "Base model:  $BASE_MODEL"
echo "Train data:  $TRAIN_DATA"
echo "Output dir:  $OUTPUT_DIR"
echo "Cache dir:   $CACHE_DIR"
echo "QLoRA:       enabled"
echo "============================================"

CMD="CUDA_VISIBLE_DEVICES=0,1 python training/dpo_training.py \
    --model_name_or_path $BASE_MODEL \
    --train_file_dir $TRAIN_DATA \
    --validation_file_dir $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
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
    --preprocessing_num_workers $PREPROCESS_WORKERS \
    --cache_dir $CACHE_DIR \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --remove_unused_columns False \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES"

if $DRY_RUN; then
    echo "[DRY RUN] Command:"
    echo "$CMD" | sed 's/    /  /g'
else
    eval "$CMD"
fi

echo ""
echo "============================================"
echo "7B DPO training complete."
echo "============================================"
