#!/bin/bash
#
# Run SFT A/B/C training for a 7B model with conservative QLoRA settings.
#
# Usage:
#   bash scripts/experiments/run_sft_training_7b.sh
#   bash scripts/experiments/run_sft_training_7b.sh --group C
#   bash scripts/experiments/run_sft_training_7b.sh --dry_run
#

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cache}"
TRAIN_GROUPS="A B C"
DRY_RUN=false

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LEARNING_RATE="1e-5"
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_LENGTH=512
WARMUP_STEPS=5
WEIGHT_DECAY=0.05
SEED=42
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10
EVAL_STEPS=25
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-10}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-2}"

TARGET_MODULES="${TARGET_MODULES:-all}"
ENABLE_FLASH_ATTN="${ENABLE_FLASH_ATTN:-auto}"
ACTIVATE_ENV="${ACTIVATE_ENV:-true}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-medicalgpt}"

# ── Group configs ──
declare -A GROUP_DATA
GROUP_DATA[A]="data/experiments/sft/sft_A_medical_only"
GROUP_DATA[B]="data/experiments/sft/sft_B_medical_general_1_1"
GROUP_DATA[C]="data/experiments/sft/sft_C_clean_1_1"

declare -A GROUP_OUTPUT
GROUP_OUTPUT[A]="outputs/experiments/models/sft_7b_A"
GROUP_OUTPUT[B]="outputs/experiments/models/sft_7b_B"
GROUP_OUTPUT[C]="outputs/experiments/models/sft_7b_C"

while [[ $# -gt 0 ]]; do
    case $1 in
        --group) TRAIN_GROUPS="$2"; shift 2 ;;
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

FLASH_ATTN_FLAG=""
if [[ "$ENABLE_FLASH_ATTN" == "true" ]]; then
    FLASH_ATTN_FLAG="--flash_attn True"
elif [[ "$ENABLE_FLASH_ATTN" == "auto" ]]; then
    if python -c "import flash_attn" >/dev/null 2>&1; then
        FLASH_ATTN_FLAG="--flash_attn True"
    fi
fi

echo "============================================"
echo "SFT 7B Conservative Training"
echo "Base model: $BASE_MODEL"
echo "Cache dir:  $CACHE_DIR"
echo "Groups:     $TRAIN_GROUPS"
echo "QLoRA:      enabled"
echo "============================================"

for group in $TRAIN_GROUPS; do
    DATA_DIR="${GROUP_DATA[$group]}"
    OUTPUT_DIR="${GROUP_OUTPUT[$group]}"

    if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
        echo "[ERROR] Training data not found: $DATA_DIR/train.jsonl"
        exit 1
    fi

    mkdir -p "$(dirname "$OUTPUT_DIR")" "$CACHE_DIR"

    echo ""
    echo "────────────────────────────────────────────"
    echo "  SFT-7B-$group"
    echo "  Data:   $DATA_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo "────────────────────────────────────────────"

    CMD="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/supervised_finetuning.py \
        --model_name_or_path $BASE_MODEL \
        --train_file_dir $DATA_DIR \
        --validation_file_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --use_peft True \
        --qlora True \
        --load_in_4bit True \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules $TARGET_MODULES \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --model_max_length $MAX_LENGTH \
        --warmup_steps $WARMUP_STEPS \
        --weight_decay $WEIGHT_DECAY \
        --seed $SEED \
        --save_steps $SAVE_STEPS \
        --save_strategy steps \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --logging_steps $LOGGING_STEPS \
        --logging_strategy steps \
        --eval_steps $EVAL_STEPS \
        --eval_strategy steps \
        --do_train \
        --do_eval \
        --bf16 True \
        --torch_dtype bfloat16 \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False \
        --report_to tensorboard \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --max_eval_samples $MAX_EVAL_SAMPLES \
        --preprocessing_num_workers $PREPROCESS_WORKERS \
        --cache_dir $CACHE_DIR \
        $FLASH_ATTN_FLAG"

    if $DRY_RUN; then
        echo "[DRY RUN] Command:"
        echo "$CMD" | sed 's/        /  /g'
    else
        eval "$CMD"
    fi
done

echo ""
echo "============================================"
echo "7B SFT training complete."
echo "============================================"
