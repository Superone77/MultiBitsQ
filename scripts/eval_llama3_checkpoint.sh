#!/bin/bash
# Script to evaluate checkpoints during training
# Usage: ./eval_llama3_checkpoint.sh [EXP_NAME] [CHECKPOINT_PATH]
#   - If EXP_NAME is provided, evaluates all checkpoints in that experiment
#   - If CHECKPOINT_PATH is provided, evaluates only that specific checkpoint

source /fast/sliu/envs/multibitsq_env/bin/activate
WORK_DIR="/fast/sliu/kwang/"

# Default configuration (matching 0_asym_llama3_1b.sh)
EXP_NAME="${1:-llama3_1b_234bit_12wsteps_bs128_lr2e-5_wonoise_asym}"
SPECIFIC_CHECKPOINT="${2:-}"
BIT_LIST="2,3,4"
EVAL_BITS_LIST="2,3,4"
EVAL_BATCH_SIZE=8
DISABLE_CLIPVALS=True
CONTAIN_WEIGHT_CLIP_VAL=False

# Wandb configuration
export WANDB_API_KEY=799de0cca57925184d04f4c0b6588ff554c3b9ec
export WANDB_PROJECT="MultiBitsQ"
export WANDB_ENTITY="yangwq177-qti"
export WANDB_RUN_NAME="${EXP_NAME}_checkpoint_eval"

echo "========================================="
echo "Checkpoint Evaluation Script"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo "Experiment Name: $EXP_NAME"
echo ""

cd $WORK_DIR

echo "[Step 1/3] Setting up environment..."
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate multibitsq_env
# echo "✓ Environment activated"
echo ""

cd $WORK_DIR/MultiBitsQ/ParetoQ

# Function to evaluate a checkpoint
evaluate_checkpoint() {
    local CHECKPOINT_PATH=$1
    local CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)
    
    echo ""
    echo "Evaluating checkpoint: $CHECKPOINT_NAME"
    echo "========================================="
    
    echo "Evaluation configuration:"
    echo "  - Checkpoint: $CHECKPOINT_NAME"
    echo "  - Checkpoint path: $CHECKPOINT_PATH"
    echo "  - Eval data: wikitext_10k_samples.jsonl"
    echo "  - w_bits_list: $BIT_LIST"
    echo "  - eval_bit_list: $EVAL_BITS_LIST"
    echo ""
    
    torchrun --nnodes=1 --nproc_per_node=2 train.py \
    --local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
    --input_model_filename "$CHECKPOINT_PATH" \
    --output_model_filename "${CHECKPOINT_NAME}_eval" \
    --train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
    --eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
    --do_train False \
    --do_eval True \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/eval_log/${CHECKPOINT_NAME}/" \
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
--w_bits_list $BIT_LIST \
    --eval_bit_list $EVAL_BITS_LIST \
    --contain_weight_clip_val $CONTAIN_WEIGHT_CLIP_VAL \
    
    echo "✓ Checkpoint $CHECKPOINT_NAME evaluation completed"
}

echo "[Step 2/3] Finding checkpoints..."
CHECKPOINT_DIR="$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/checkpoint/"

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
            
            for CHECKPOINT in $CHECKPOINTS; do
                evaluate_checkpoint "$CHECKPOINT"
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
echo "[Step 3/3] Summary"
echo "========================================="
echo "Checkpoint evaluation completed!"
echo "Results saved to: $WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/eval_log/"
echo "========================================="

