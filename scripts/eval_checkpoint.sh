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

# If EXP_NAME is not provided, try to find the latest experiment
if [ -z "$EXP_NAME" ]; then
    echo "[Step 2/4] Finding latest experiment..."
    if [ -d "$SAVE_DIR/MultiBitsQ/tmp" ]; then
        # Find the most recently modified experiment directory
        EXP_NAME=$(find "$SAVE_DIR/MultiBitsQ/tmp" -maxdepth 1 -type d -name "*" ! -name "tmp" | sort -t/ -k6 -r | head -1 | xargs basename)
        if [ -z "$EXP_NAME" ]; then
            echo "Error: No experiments found in $SAVE_DIR/MultiBitsQ/tmp"
            exit 1
        fi
        echo "Using latest experiment: $EXP_NAME"
    else
        echo "Error: Experiment directory not found: $SAVE_DIR/MultiBitsQ/tmp"
        echo "Please provide EXP_NAME as first argument"
        exit 1
    fi
else
    echo "[Step 2/4] Using provided experiment: $EXP_NAME"
fi

# Set WANDB_RUN_NAME
export WANDB_RUN_NAME="${EXP_NAME}_checkpoint_eval"

# Try to detect model type from experiment name or checkpoint
# This determines share_embedding and layer_sharing settings
# MobileLLM models support share_embedding and layer_sharing
# Llama and other models do not
EXP_NAME_LOWER=$(echo "$EXP_NAME" | tr '[:upper:]' '[:lower:]')
if [[ "$EXP_NAME_LOWER" == *"mobilellm"* ]]; then
    SHARE_EMBEDDING="True"
    LAYER_SHARING="True"
    MODEL_TYPE="MobileLLM"
    echo "Detected MobileLLM model - enabling share_embedding and layer_sharing"
elif [[ "$EXP_NAME_LOWER" == *"llama"* ]]; then
    SHARE_EMBEDDING="False"
    LAYER_SHARING="False"
    MODEL_TYPE="Llama"
    echo "Detected Llama model - disabling share_embedding and layer_sharing"
else
    # Try to detect from checkpoint config.json if available
    CHECKPOINT_DIR="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/"
    FIRST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | head -1)
    if [ -n "$FIRST_CHECKPOINT" ] && [ -f "$FIRST_CHECKPOINT/config.json" ]; then
        # Try to read model type from config.json
        if grep -q "MobileLLM" "$FIRST_CHECKPOINT/config.json" 2>/dev/null; then
            SHARE_EMBEDDING="True"
            LAYER_SHARING="True"
            MODEL_TYPE="MobileLLM (detected from config)"
            echo "Detected MobileLLM model from config.json - enabling share_embedding and layer_sharing"
        else
            SHARE_EMBEDDING="False"
            LAYER_SHARING="False"
            MODEL_TYPE="Unknown (defaulting to False)"
            echo "Could not detect model type from EXP_NAME, defaulting to share_embedding=False, layer_sharing=False"
        fi
    else
        SHARE_EMBEDDING="False"
        LAYER_SHARING="False"
        MODEL_TYPE="Unknown (defaulting to False)"
        echo "Could not detect model type, defaulting to share_embedding=False, layer_sharing=False"
    fi
fi

echo ""
echo "========================================="
echo "Checkpoint Evaluation Script"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo "Save Directory: $SAVE_DIR"
echo "Experiment Name: $EXP_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Configuration:"
echo "  - BIT_LIST: $BIT_LIST"
echo "  - EVAL_BITS_LIST: $EVAL_BITS_LIST"
echo "  - EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
echo "  - GPU_NUM: $GPU_NUM"
echo "  - Share Embedding: $SHARE_EMBEDDING"
echo "  - Layer Sharing: $LAYER_SHARING"
echo ""

# Check and change to ParetoQ directory
if [ ! -d "$WORK_DIR/MultiBitsQ/ParetoQ" ]; then
    echo "Error: ParetoQ directory does not exist: $WORK_DIR/MultiBitsQ/ParetoQ"
    exit 1
fi
cd "$WORK_DIR/MultiBitsQ/ParetoQ" || exit 1

# Function to evaluate a checkpoint
evaluate_checkpoint() {
    local CHECKPOINT_PATH=$1
    local CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
    
    echo ""
    echo "Evaluating checkpoint: $CHECKPOINT_NAME"
    echo "========================================="
    
    echo "Evaluation configuration:"
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

echo "[Step 3/4] Finding checkpoints..."
CHECKPOINT_DIR="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/"

if [ -n "$SPECIFIC_CHECKPOINT" ]; then
    # Evaluate specific checkpoint
    if [ -d "$SPECIFIC_CHECKPOINT" ]; then
        echo "Evaluating specific checkpoint: $SPECIFIC_CHECKPOINT"
        evaluate_checkpoint "$SPECIFIC_CHECKPOINT"
    elif [ -d "$CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT" ]; then
        echo "Evaluating specific checkpoint: $CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT"
        evaluate_checkpoint "$CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT"
    else
        echo "Error: Checkpoint not found: $SPECIFIC_CHECKPOINT"
        echo "  Tried: $SPECIFIC_CHECKPOINT"
        echo "  Tried: $CHECKPOINT_DIR/$SPECIFIC_CHECKPOINT"
        exit 1
    fi
else
    # Evaluate all checkpoints
    if [ -d "$CHECKPOINT_DIR" ]; then
        # Sort checkpoints by step number (checkpoint-XXXXX format)
        CHECKPOINTS=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
        
        if [ -z "$CHECKPOINTS" ]; then
            echo "No checkpoints found in $CHECKPOINT_DIR"
            exit 1
        else
            CHECKPOINT_COUNT=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')
            echo "Found $CHECKPOINT_COUNT checkpoint(s) to evaluate"
            echo ""
            
            # Use while read to handle paths with spaces properly
            echo "$CHECKPOINTS" | while IFS= read -r CHECKPOINT; do
                if [ -n "$CHECKPOINT" ] && [ -d "$CHECKPOINT" ]; then
                    evaluate_checkpoint "$CHECKPOINT"
                elif [ -n "$CHECKPOINT" ]; then
                    echo "Warning: Checkpoint directory not found: $CHECKPOINT"
                fi
            done
            
            echo ""
            echo "✓ All checkpoint evaluations completed"
        fi
    else
        echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
fi

echo ""
echo "[Step 4/4] Summary"
echo "========================================="
echo "Checkpoint evaluation completed!"
echo "Results saved to: $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/eval_log/"
echo "========================================="

