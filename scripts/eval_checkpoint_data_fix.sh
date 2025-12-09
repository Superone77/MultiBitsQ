#!/bin/bash
# Script to evaluate checkpoints during training
# Supports both MobileLLM and Llama models
# 
# Usage: ./eval_checkpoint.sh [EXP_NAME] [CHECKPOINT_PATH]
#   - If EXP_NAME is provided, evaluates all checkpoints in that experiment
#   - If CHECKPOINT_PATH is provided, evaluates only that specific checkpoint
#   - If neither is provided, will try to find the latest experiment
#
# Examples:
#   # Evaluate all checkpoints for a MobileLLM experiment
#   ./eval_checkpoint.sh mobilellm_125m_234bit_24wsteps_bs64_lr2e-4_wonoise
#
#   # Evaluate all checkpoints for a Llama experiment
#   ./eval_checkpoint.sh llama_3___2_1b_234bit_24wsteps_bs32_lr2e-4_wonoise
#
#   # Evaluate a specific checkpoint
#   ./eval_checkpoint.sh mobilellm_125m_234bit_24wsteps_bs64_lr2e-4_wonoise checkpoint-20000
#
#   # Auto-detect latest experiment
#   ./eval_checkpoint.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"

# Activate virtual environment
echo "[Step 1/4] Activating environment..."
if [ -f "/fast/wangk/virtual_env/multibitsq_env/bin/activate" ]; then
    source /fast/wangk/virtual_env/multibitsq_env/bin/activate
else 
    echo "Error: Virtual environment not found at /fast/wangk/virtual_env/multibitsq_env/bin/activate"
    exit 1
fi
echo "✓ Environment activated"
echo ""

# Parse arguments
EXP_NAME="${1:-}"
SPECIFIC_CHECKPOINT="${2:-}"

# Default configuration (can be overridden by EXP_NAME or environment)
BIT_LIST="${BIT_LIST:-2,3,4}"
EVAL_BITS_LIST="${EVAL_BITS_LIST:-2,3,4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DISABLE_CLIPVALS="${DISABLE_CLIPVALS:-False}"
CONTAIN_WEIGHT_CLIP_VAL="${CONTAIN_WEIGHT_CLIP_VAL:-True}"
GPU_NUM="${GPU_NUM:-8}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-/fast/wangk/MultiBitsQ/train_data/finewebedu_6000k_samples.jsonl}"
EVAL_DATA="${EVAL_DATA:-/fast/wangk/MultiBitsQ/eval_data/wikitext_10k_samples.jsonl}"

# Wandb configuration
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=799de0cca57925184d04f4c0b6588ff554c3b9ec
fi
export WANDB_PROJECT="MultiBitsQ"
export WANDB_ENTITY="yangwq177-qti"

# Define experiments to evaluate (if no arguments provided)
if [ -z "$EXP_NAME" ]; then
    # Default experiments to evaluate
    EXP_NAMES=(
        "llama_3___2_1B_234bit_24wsteps_bs32_lr2e-4_wonoise_20251208"
        "f3f50d38B7ccbbe64B3ec415313478ce7Be1a8ef_234bit_24wsteps_bs64_lr2e-4_wonoise_20251208"
    )
    echo "[Step 2/4] Using default experiments to evaluate:"
    for exp in "${EXP_NAMES[@]}"; do
        echo "  - $exp"
    done
else
    # Single experiment mode
    EXP_NAMES=("$EXP_NAME")
    echo "[Step 2/4] Using provided experiment: $EXP_NAME"
fi

# Check and change to ParetoQ directory
if [ ! -d "$WORK_DIR/MultiBitsQ/ParetoQ" ]; then
    echo "Error: ParetoQ directory does not exist: $WORK_DIR/MultiBitsQ/ParetoQ"
    exit 1
fi
cd "$WORK_DIR/MultiBitsQ/ParetoQ" || exit 1

# Function to detect model type for an experiment
detect_model_type() {
    local EXP_NAME=$1
    local EXP_NAME_LOWER=$(echo "$EXP_NAME" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$EXP_NAME_LOWER" == *"mobilellm"* ]]; then
        echo "MobileLLM"
    elif [[ "$EXP_NAME_LOWER" == *"llama"* ]]; then
        echo "Llama"
    else
        # Try to detect from checkpoint config.json if available
        local CHECKPOINT_DIR="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/"
        local FIRST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | head -1)
        if [ -n "$FIRST_CHECKPOINT" ] && [ -f "$FIRST_CHECKPOINT/config.json" ]; then
            if grep -q "MobileLLM" "$FIRST_CHECKPOINT/config.json" 2>/dev/null; then
                echo "MobileLLM"
            else
                echo "Unknown"
            fi
        else
            echo "Unknown"
        fi
    fi
}

# Function to get model settings based on model type
get_model_settings() {
    local MODEL_TYPE=$1
    if [ "$MODEL_TYPE" = "MobileLLM" ]; then
        echo "True True"  # SHARE_EMBEDDING LAYER_SHARING
    else
        echo "False False"  # SHARE_EMBEDDING LAYER_SHARING
    fi
}

# Function to evaluate a checkpoint
evaluate_checkpoint() {
    local EXP_NAME=$1
    local CHECKPOINT_PATH=$2
    local CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
    local SHARE_EMBEDDING=$3
    local LAYER_SHARING=$4
    
    echo ""
    echo "Evaluating checkpoint: $CHECKPOINT_NAME"
    echo "========================================="
    
    echo "Evaluation configuration:"
    echo "  - Experiment: $EXP_NAME"
    echo "  - Checkpoint: $CHECKPOINT_NAME"
    echo "  - Checkpoint path: $CHECKPOINT_PATH"
    echo "  - Eval data: $(basename $EVAL_DATA)"
    echo "  - w_bits_list: $BIT_LIST"
    echo "  - eval_bit_list: $EVAL_BITS_LIST"
    echo ""
    
    if ! torchrun --nnodes=1 --nproc_per_node=$GPU_NUM train.py \
    --local_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/" \
    --input_model_filename "$CHECKPOINT_PATH" \
    --output_model_filename "${CHECKPOINT_NAME}_eval" \
    --train_data_local_path "$TRAIN_DATA" \
    --eval_data_local_path "$EVAL_DATA" \
    --do_train False \
    --do_eval True \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --use_lm_eval True \
    --lm_eval_tasks wikitext \
    --log_on_each_node False \
    --logging_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/eval_log/${CHECKPOINT_NAME}/" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --report_to "wandb" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing False \
    --qat True \
    --w_bits 2 \
    --w_bits_list $BIT_LIST \
    --eval_bit_list $EVAL_BITS_LIST \
    --contain_weight_clip_val $CONTAIN_WEIGHT_CLIP_VAL \
    --multiple_bits_disable_clipvals $DISABLE_CLIPVALS \
    --share_embedding $SHARE_EMBEDDING \
    --layer_sharing $LAYER_SHARING; then
        echo "Error: Checkpoint evaluation failed for $CHECKPOINT_NAME"
        return 1
    fi 
    
    echo "✓ Checkpoint $CHECKPOINT_NAME evaluation completed"
}

# Function to evaluate all checkpoints for an experiment
evaluate_experiment() {
    local EXP_NAME=$1
    local CHECKPOINT_DIR="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/"
    
    echo ""
    echo "========================================="
    echo "Evaluating Experiment: $EXP_NAME"
    echo "========================================="
    
    # Detect model type
    MODEL_TYPE=$(detect_model_type "$EXP_NAME")
    MODEL_SETTINGS=($(get_model_settings "$MODEL_TYPE"))
    SHARE_EMBEDDING="${MODEL_SETTINGS[0]}"
    LAYER_SHARING="${MODEL_SETTINGS[1]}"
    
    echo "Model Type: $MODEL_TYPE"
    echo "Configuration:"
    echo "  - BIT_LIST: $BIT_LIST"
    echo "  - EVAL_BITS_LIST: $EVAL_BITS_LIST"
    echo "  - EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
    echo "  - GPU_NUM: $GPU_NUM"
    echo "  - Share Embedding: $SHARE_EMBEDDING"
    echo "  - Layer Sharing: $LAYER_SHARING"
    echo ""
    
    if [ -n "$SPECIFIC_CHECKPOINT" ]; then
        # Evaluate specific checkpoint
        if [ -d "$SPECIFIC_CHECKPOINT" ]; then
            echo "Evaluating specific checkpoint: $SPECIFIC_CHECKPOINT"
            evaluate_checkpoint "$EXP_NAME" "$SPECIFIC_CHECKPOINT" "$SHARE_EMBEDDING" "$LAYER_SHARING"
        elif [ -d "$CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT" ]; then
            echo "Evaluating specific checkpoint: $CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT"
            evaluate_checkpoint "$EXP_NAME" "$CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT" "$SHARE_EMBEDDING" "$LAYER_SHARING"
        else
            echo "Error: Checkpoint not found: $SPECIFIC_CHECKPOINT"
            echo "  Tried: $SPECIFIC_CHECKPOINT"
            echo "  Tried: $CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT"
            return 1
        fi
    else
        # Evaluate all checkpoints
        if [ -d "$CHECKPOINT_DIR" ]; then
            # Sort checkpoints by step number (checkpoint-XXXXX format)
            CHECKPOINTS=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
            
            if [ -z "$CHECKPOINTS" ]; then
                echo "No checkpoints found in $CHECKPOINT_DIR"
                return 1
            else
                CHECKPOINT_COUNT=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')
                echo "Found $CHECKPOINT_COUNT checkpoint(s) to evaluate"
                echo ""
                
                # Use while read to handle paths with spaces properly
                echo "$CHECKPOINTS" | while IFS= read -r CHECKPOINT; do
                    if [ -n "$CHECKPOINT" ] && [ -d "$CHECKPOINT" ]; then
                        evaluate_checkpoint "$EXP_NAME" "$CHECKPOINT" "$SHARE_EMBEDDING" "$LAYER_SHARING"
                    elif [ -n "$CHECKPOINT" ]; then
                        echo "Warning: Checkpoint directory not found: $CHECKPOINT"
                    fi
                done
                
                echo ""
                echo "✓ All checkpoint evaluations completed for $EXP_NAME"
            fi
        else
            echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
            return 1
        fi
    fi
}

echo ""
echo "========================================="
echo "Checkpoint Evaluation Script"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo "Save Directory: $SAVE_DIR"
echo "Number of experiments: ${#EXP_NAMES[@]}"
echo ""

echo "[Step 3/4] Starting evaluation for all experiments..."
# Evaluate all experiments
for EXP_NAME in "${EXP_NAMES[@]}"; do
    evaluate_experiment "$EXP_NAME"
done

echo ""
echo "[Step 4/4] Summary"
echo "========================================="
echo "Checkpoint evaluation completed for all experiments!"
for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo "Results for $EXP_NAME saved to: $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/eval_log/"
done
echo "========================================="

