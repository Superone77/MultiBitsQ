#!/bin/bash
# add noise injection and pre-quantization noise to the training
# Parent Directory of MultiBitQ，e.g. /fast/sliu/wanqi
set -e  # Exit on error
set -u  # Exit on undefined variable


WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"
cd $SAVE_DIR
# Activate virtual environment (check if exists)

# export MODELSCOPE_DISABLE_LOCK=true

echo "[Step 1/6] activate environment"
if [ -f "/fast/wangk/virtual_env/multibitsq_env/bin/activate" ]; then
    source /fast/wangk/virtual_env/multibitsq_env/bin/activate
else 
    echo "Installing requirements..."
    source ~/miniforge3/etc/profile.d/conda.sh
    conda create -n multibitsq_env python=3.10 -y
    conda activate multibitsq_env
    pip install -r $WORK_DIR/MultiBitsQ/ParetoQ/requirement.txt
    echo "✓ Requirements installed"
    echo ""
fi
echo "[Step 1/6] environment activated"




# export HF_TOKEN=<>
INPUT_MODEL="$SAVE_DIR/MultiBitsQ/model/LLM-Research/Llama-3___2-1B"
# Update these paths if needed to match your actual data locations
TRAIN_DATA="/fast/wangk/MultiBitsQ/train_data/finewebedu_train_samples.jsonl"
EVAL_DATA="/fast/wangk/MultiBitsQ/eval_data/wikitext_10k_samples.jsonl"
CACHE_DIR="/fast/wangk/MultiBitsQ/cache/pretokenized"
BIT_LIST="1,2,4"
PROB_LIST="2,2,1"
EVAL_BITS_LIST="1,2,4"
MAX_STEPS=40000
BATCH_SIZE=8
ACCU_STEP=2
LEARNING_RATE=2e-5
CONTAIN_WEIGHT_CLIP_VAL=True
RANDOM_ASSIGN=True
RANDOM_ASSIGN_PROB=0.3
EVAL_BATCH_SIZE=8
OUTPUT_MODEL_FILENAME="llama3_1B-multibitq"
GPU_NUM=8

# Auto-generate EXP_NAME from configuration
# Extract model name from INPUT_MODEL path (e.g., MobileLLM-125M -> mobilellm_125M)
MODEL_BASE=$(basename "$INPUT_MODEL")
# Convert to lowercase and replace hyphens with underscores
MODEL_TEMP=$(echo "$MODEL_BASE" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g')
# Capitalize size suffix (M, B, etc.) after numbers
MODEL_NAME=$(echo "$MODEL_TEMP" | sed 's/\([0-9]\)m/\1M/g' | sed 's/\([0-9]\)b/\1B/g')
# Extract bit list without commas (e.g., 2,3,4 -> 234)
BIT_STR=$(echo "$BIT_LIST" | tr -d ',')
# Calculate total batch size
TOTAL_BS=$((GPU_NUM * BATCH_SIZE * ACCU_STEP))
# Convert MAX_STEPS to "wsteps" format (e.g., 240000 -> 24wsteps)
if [ $MAX_STEPS -ge 10000 ]; then
    WSTEPS=$((MAX_STEPS / 10000))
    STEPS_STR="${WSTEPS}wsteps"
else
    STEPS_STR="${MAX_STEPS}steps"
fi
# Format learning rate (e.g., 2e-4 -> lr2e-4)
LR_STR=$(echo "$LEARNING_RATE" | sed 's/^/lr/')
# Format probability configs for safe filenames (e.g., 0.5 -> 0p5, 0.2,0.3,0.5 -> 0p2x0p3x0p5)
RAP_STR=$(echo "$RANDOM_ASSIGN_PROB" | sed 's/\./p/g')
PROB_STR=$(echo "$PROB_LIST" | tr -d ' ' | sed 's/\./p/g' | sed 's/,/x/g')
# Get current date (e.g., 20240101)
DATE_STR=$(date +%Y%m%d)

# Generate EXP_NAME
EXP_NAME="${MODEL_NAME}_${BIT_STR}bit_${STEPS_STR}_bs${TOTAL_BS}_${LR_STR}_rap${RAP_STR}_prob${PROB_STR}_${DATE_STR}"


rm -rf $CACHE_DIR
# Wandb configuration
# Note: Consider using environment variables or a config file for API keys
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=799de0cca57925184d04f4c0b6588ff554c3b9ec
fi
export WANDB_PROJECT="MultiBitsQ"
export WANDB_ENTITY="yangwq177-qti"
export WANDB_RUN_NAME="$EXP_NAME"

echo "========================================="
echo "Starting Multi-Bit Training Pipeline"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo "Experiment Name: $EXP_NAME"
echo ""

echo "[Step 2/6] Creating directories..."
mkdir -p $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/
mkdir -p $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/log/
mkdir -p $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/
mkdir -p $SAVE_DIR/MultiBitsQ/model/
mkdir -p $SAVE_DIR/MultiBitsQ/eval_data/
mkdir -p $SAVE_DIR/MultiBitsQ/train_data/
echo "✓ Directories created"
echo ""

# Check and change to work directory
if [ ! -d "$SAVE_DIR" ]; then
    echo "Error: Work directory does not exist: $SAVE_DIR"
    exit 1
fi
cd "$SAVE_DIR" || exit 1



echo "[Step 3/6] Downloading data and models..."
# Check if WORK_DIR/MultiBitsQ exists
if [ ! -d "$WORK_DIR/MultiBitsQ" ]; then
    echo "Error: $WORK_DIR/MultiBitsQ does not exist!"
    exit 1
fi

# Download training data (uncomment if needed)
# python $WORK_DIR/MultiBitsQ/scripts/download_data.py --output_dir $SAVE_DIR/MultiBitsQ/train_data/

# Download models
# if ! python $WORK_DIR/MultiBitsQ/scripts/download_model.py --output_dir $SAVE_DIR/MultiBitsQ/model/ --models ; then
#     echo "Error: Failed to download models"
#     exit 1
# fi

# Download evaluation data
# if ! python $WORK_DIR/MultiBitsQ/scripts/download_wiki.py --output_dir $SAVE_DIR/MultiBitsQ/eval_data/; then
#     echo "Error: Failed to download evaluation data"
#     exit 1
# fi

# Verify required files exist
MISSING_FILES=0
if [ ! -f "$INPUT_MODEL/config.json" ]; then
    echo "Warning: Input model not found at $INPUT_MODEL"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Warning: Training data not found at $TRAIN_DATA"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ ! -f "$EVAL_DATA" ]; then
    echo "Warning: Evaluation data not found at $EVAL_DATA"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo "Warning: $MISSING_FILES required file(s) are missing. The script will continue but may fail later."
    echo ""
fi

echo "✓ Data and models downloaded"
echo ""

# Check and change to ParetoQ directory
if [ ! -d "$WORK_DIR/MultiBitsQ/ParetoQ" ]; then
    echo "Error: ParetoQ directory does not exist: $WORK_DIR/MultiBitsQ/ParetoQ"
    exit 1
fi
cd "$WORK_DIR/MultiBitsQ/ParetoQ" || exit 1

echo "[Step 4/6] Starting training phase..."
echo "Training configuration:"
echo "  - INPUT_MODEL: $INPUT_MODEL"
echo "  - Multi-bit: $BIT_LIST"
echo "  - Probability: $PROB_LIST"
echo "  - Max steps: $MAX_STEPS"
echo "  - Batch size: $BATCH_SIZE (per device)"
echo "  - Random Assign: $RANDOM_ASSIGN"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

if ! torchrun --nnodes=1 --nproc_per_node=$GPU_NUM train.py \
--local_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/" \
--output_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/" \
--input_model_filename "$INPUT_MODEL" \
--output_model_filename "$OUTPUT_MODEL_FILENAME" \
--train_data_local_path "$TRAIN_DATA" \
--eval_data_local_path "$EVAL_DATA" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/log" \
--max_steps $MAX_STEPS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps $ACCU_STEP \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "wandb" \
--save_total_limit 24 \
--learning_rate $LEARNING_RATE \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 200 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits_list $BIT_LIST \
--prob_list $PROB_LIST \
--multiple_bits_random_assign $RANDOM_ASSIGN \
--multiple_bits_random_assign_prob $RANDOM_ASSIGN_PROB ; then
    echo "Error: Training failed!"
    exit 1
fi 


# Model is saved to local_dir/models/output_model_filename
# local_dir is set to $WORK_DIR/MultiBitsQ/tmp/$EXP_NAME/
OUTPUT_MODEL="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/models/$OUTPUT_MODEL_FILENAME"
echo ""
if [ ! -d "$OUTPUT_MODEL" ]; then
    echo "Warning: Output model directory not found at $OUTPUT_MODEL"
    echo "Training may have failed or model was saved to a different location"
else
    echo "✓ Training completed"
    echo "  - OUTPUT_MODEL: $OUTPUT_MODEL"
fi
echo ""

echo "[Step 5/6] Starting evaluation phase..."


echo "Evaluation configuration:"
echo "  - Model: $OUTPUT_MODEL"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits_list: $BIT_LIST"
echo "  - eval_bit_list: $EVAL_BITS_LIST"
echo ""

if ! torchrun --nnodes=1 --nproc_per_node=$GPU_NUM train.py \
--local_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/" \
--input_model_filename "$OUTPUT_MODEL" \
--output_model_filename "mobilellm125M-1234bit_eval" \
--train_data_local_path "$TRAIN_DATA" \
--eval_data_local_path "$EVAL_DATA" \
--do_train False \
--do_eval True \
--use_lm_eval True \
--lm_eval_tasks "wikitext" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/eval_log/" \
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
--contain_weight_clip_val $CONTAIN_WEIGHT_CLIP_VAL; then
    echo "Error: Evaluation failed!"
    exit 1
fi 

echo ""
echo "✓ Evaluation completed"
echo ""

echo "[Step 6/6] Starting checkpoint evaluation phase..."
CHECKPOINT_DIR="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/"

# Function to evaluate a checkpoint with different w_bits
evaluate_checkpoint() {
    local CHECKPOINT_PATH=$1
    local CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
    
    echo ""
    echo "Evaluating checkpoint: $CHECKPOINT_NAME"
    echo "========================================="
    
    echo "Evaluation configuration:"
    echo "  - Checkpoint: $CHECKPOINT_NAME"
    echo "  - Eval data: wikitext_10k_samples.jsonl"
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
    --use_lm_eval True \
    --lm_eval_tasks "wikitext" \
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
    --w_bits_list $BIT_LIST \
    --eval_bit_list $EVAL_BITS_LIST \
    --contain_weight_clip_val $CONTAIN_WEIGHT_CLIP_VAL; then
        echo "Error: Checkpoint evaluation failed for $CHECKPOINT_NAME"
        return 1
    fi 
    
    echo "✓ Checkpoint $CHECKPOINT_NAME evaluation completed"
}

# Find and evaluate all checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    # Sort checkpoints by step number (checkpoint-XXXXX format)
    CHECKPOINTS=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
    
    if [ -z "$CHECKPOINTS" ]; then
        echo "No checkpoints found in $CHECKPOINT_DIR"
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
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
fi

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="