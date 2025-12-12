#!/bin/bash
# 2-bit training entry for the reference/ParetoQ implementation.
set -e
set -u

WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"

echo "[Step 1/4] activate environment"
if [ -f "/fast/wangk/virtual_env/multibitsq_env/bin/activate" ]; then
    source /fast/wangk/virtual_env/multibitsq_env/bin/activate
else
    echo "Installing requirements..."
    source ~/miniforge3/etc/profile.d/conda.sh
    conda create -n multibitsq_env python=3.10 -y
    conda activate multibitsq_env
    pip install -r "$WORK_DIR/MultiBitsQ/ParetoQ/requirement.txt"
    echo "✓ Requirements installed"
fi
echo "[Step 1/4] environment ready"

INPUT_MODEL="$SAVE_DIR/MultiBitsQ/model/LLM-Research/Llama-3___2-1B"
TRAIN_DATA="$SAVE_DIR/MultiBitsQ/train_data/finewebedu_train_samples.jsonl"
EVAL_DATA="$SAVE_DIR/MultiBitsQ/eval_data/wikitext_10k_samples.jsonl"
CACHE_DIR="$SAVE_DIR/MultiBitsQ/cache/pretokenized"
GPU_NUM=8
BATCH_SIZE=8
ACCU_STEP=2
NUM_EPOCHS=1
LEARNING_RATE=2e-5
OUTPUT_MODEL_FILENAME="llama3_1B_2bit"

# Build experiment name similar to 0_Llama3_1B_lr_1e-4.sh
MODEL_BASE=$(basename "$INPUT_MODEL")
MODEL_TEMP=$(echo "$MODEL_BASE" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g')
MODEL_NAME=$(echo "$MODEL_TEMP" | sed 's/\([0-9]\)m/\1M/g' | sed 's/\([0-9]\)b/\1B/g')
TOTAL_BS=$((GPU_NUM * BATCH_SIZE * ACCU_STEP))
LR_STR=$(echo "$LEARNING_RATE" | sed 's/^/lr/')
DATE_STR=$(date +%Y%m%d)
EXP_NAME="${MODEL_NAME}_2bit_${NUM_EPOCHS}ep_bs${TOTAL_BS}_${LR_STR}_${DATE_STR}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=799de0cca57925184d04f4c0b6588ff554c3b9ec
fi
export WANDB_PROJECT="MultiBitsQ"
export WANDB_ENTITY="yangwq177-qti"
export WANDB_RUN_NAME="$EXP_NAME"

echo "========================================="
echo "Starting ParetoQ 2-bit training"
echo "  WORK_DIR: $WORK_DIR"
echo "  SAVE_DIR: $SAVE_DIR"
echo "  EXP_NAME: $EXP_NAME"
echo "  CACHE_DIR: $CACHE_DIR"
echo "  INPUT_MODEL: $INPUT_MODEL"
echo "========================================="

echo "[Step 2/4] prepare directories"
mkdir -p "$CACHE_DIR"
mkdir -p "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME"/{log,checkpoint}
mkdir -p "$SAVE_DIR/MultiBitsQ/model" "$SAVE_DIR/MultiBitsQ/eval_data" "$SAVE_DIR/MultiBitsQ/train_data"

if [ ! -d "$WORK_DIR/ParetoQ" ]; then
    cd $WORK_DIR/ParetoQ
    git clone https://github.com/Superone77/ParetoQ.git 
fi

MISSING_FILES=0
if [ ! -f "$INPUT_MODEL/config.json" ]; then
    echo "Warning: missing input model at $INPUT_MODEL"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Warning: missing training data at $TRAIN_DATA"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ ! -f "$EVAL_DATA" ]; then
    echo "Warning: missing eval data at $EVAL_DATA"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ $MISSING_FILES -gt 0 ]; then
    echo "Continuing despite $MISSING_FILES missing file(s)..."
fi

cd "$WORK_DIR/ParetoQ"

echo "[Step 3/4] launch training"
if ! torchrun --nnodes=1 --nproc_per_node=$GPU_NUM train.py \
--local_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/" \
--output_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/checkpoint/" \
--input_model_filename "$INPUT_MODEL" \
--output_model_filename "$OUTPUT_MODEL_FILENAME" \
--train_data_local_path "$TRAIN_DATA" \
--eval_data_local_path "$EVAL_DATA" \
--cache_dir "$CACHE_DIR" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/log" \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $ACCU_STEP \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "wandb" \
--save_total_limit 4 \
--learning_rate $LEARNING_RATE \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 20 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits 2; then
    echo "Error: 2-bit training failed!"
    exit 1
fi

OUTPUT_MODEL="$SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/models/$OUTPUT_MODEL_FILENAME"
echo "[Step 4/4] training finished"
echo "  OUTPUT_MODEL: $OUTPUT_MODEL"
echo "  LOG_DIR: $SAVE_DIR/MultiBitsQ/tmp/$EXP_NAME/log"
