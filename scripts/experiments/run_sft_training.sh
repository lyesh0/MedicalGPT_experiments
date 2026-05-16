#!/bin/bash
#
# Run SFT A/B/C ablation training with controlled hyperparameters.
#
# Usage:
#   bash scripts/experiments/run_sft_training.sh              # train all three groups
#   bash scripts/experiments/run_sft_training.sh --group A    # train SFT-A only
#   bash scripts/experiments/run_sft_training.sh --dry_run    # print commands only
#

set -euo pipefail

# ── Fixed hyperparameters (same for all groups) ──
BASE_MODEL="/root/autodl-tmp/models/Qwen3.5-2B"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LEARNING_RATE="2e-5"
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=8
MAX_LENGTH=512
WARMUP_STEPS=5
WEIGHT_DECAY=0.05
SEED=42
SAVE_STEPS=500
LOGGING_STEPS=10
EVAL_STEPS=50
MAX_TRAIN_SAMPLES=-1
MAX_EVAL_SAMPLES=20

# ── Group configs ──
declare -A GROUP_DATA
GROUP_DATA[A]="data/experiments/sft/sft_A_medical_only"
GROUP_DATA[B]="data/experiments/sft/sft_B_medical_general_1_1"
GROUP_DATA[C]="data/experiments/sft/sft_C_clean_1_1"

declare -A GROUP_OUTPUT
GROUP_OUTPUT[A]="outputs/experiments/models/sft_A"
GROUP_OUTPUT[B]="outputs/experiments/models/sft_B"
GROUP_OUTPUT[C]="outputs/experiments/models/sft_C"

# ── Parse args ──
TRAIN_GROUPS="A B C"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --group) TRAIN_GROUPS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$PROJECT_ROOT"

echo "============================================"
echo "SFT Ablation Training"
echo "Base model: $BASE_MODEL"
echo "LoRA rank: $LORA_RANK  alpha: $LORA_ALPHA"
echo "Learning rate: $LEARNING_RATE  epochs: $NUM_EPOCHS"
echo "Groups to train: $TRAIN_GROUPS"
echo "============================================"

for group in $TRAIN_GROUPS; do
    DATA_DIR="${GROUP_DATA[$group]}"
    OUTPUT_DIR="${GROUP_OUTPUT[$group]}"

    if [ ! -f "$DATA_DIR/train.jsonl" ]; then
        echo "[ERROR] Training data not found: $DATA_DIR/train.jsonl"
        echo "  Run: bash scripts/experiments/run_sft_data_pipeline.sh"
        exit 1
    fi

    echo ""
    echo "────────────────────────────────────────────"
    echo "  SFT-$group"
    echo "  Data:   $DATA_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo "────────────────────────────────────────────"

    # Check if flash_attn is available
    FLASH_ATTN_FLAG=""
    if python -c "import flash_attn" 2>/dev/null; then
        FLASH_ATTN_FLAG="--flash_attn True"
    fi

    CMD="python training/supervised_finetuning.py \
        --model_name_or_path $BASE_MODEL \
        --train_file_dir $DATA_DIR \
        --validation_file_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --use_peft True \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules all \
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
        --save_total_limit 3 \
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
        --preprocessing_num_workers 4 \
        --cache_dir ./cache \
        $FLASH_ATTN_FLAG"

    if $DRY_RUN; then
        echo "[DRY RUN] Command:"
        echo "$CMD" | sed 's/        /  /g'
    else
        echo "Training..."
        eval "$CMD"
        echo "SFT-$group done. Model saved to $OUTPUT_DIR"
    fi
done

echo ""
echo "============================================"
echo "All SFT training complete."
echo "Models:"
for group in $TRAIN_GROUPS; do
    echo "  SFT-$group -> ${GROUP_OUTPUT[$group]}"
done
echo "============================================"
echo ""
echo "Next: python tools/experiments/build_eval_set.py"
