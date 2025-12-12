#!/bin/bash
# Pretokenize script for large-scale datasets
# This script processes JSONL files and creates cached tokenized datasets
set -e  # Exit on error
set -u  # Exit on undefined variable

WORK_DIR="/home/wangk/"
SAVE_DIR="/fast/wangk/"
cd $SAVE_DIR

# Activate virtual environment (check if exists)
echo "[Step 1/4] Activating environment..."
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
echo "[Step 1/4] Environment activated"
echo ""

INPUT_MODEL="$SAVE_DIR/MultiBitsQ/model/LLM-Research/Llama-3___2-1B"
TRAIN_DATA_PATH="$SAVE_DIR/MultiBitsQ/train_data/finewebedu_train_samples.jsonl"
EVAL_DATA_PATH="$SAVE_DIR/MultiBitsQ/eval_data/wikitext_10k_samples.jsonl"
CACHE_DIR="$SAVE_DIR/MultiBitsQ/cache/pretokenized"
MODEL_MAX_LENGTH=2048
BATCH_SIZE_SAMPLES=1000

# Optional: Estimate memory requirements first
# Uncomment the following lines to estimate memory for 100B tokens:
# echo "Estimating memory requirements for 100B tokens..."
# python $WORK_DIR/MultiBitsQ/ParetoQ/pretokenize.py \
#     --input_model_filename "$INPUT_MODEL" \
#     --cache_dir "$CACHE_DIR" \
#     --train_data_local_path "$TRAIN_DATA_PATH" \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --estimate_memory \
#     --num_tokens_estimate 100000000000
# exit 0

echo "========================================="
echo "Starting Pretokenization Pipeline"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo "Save Directory: $SAVE_DIR"
echo "Input Model: $INPUT_MODEL"
echo "Train Data: $TRAIN_DATA_PATH"
echo "Eval Data: $EVAL_DATA_PATH"
echo "Cache Directory: $CACHE_DIR"
echo "Model Max Length: $MODEL_MAX_LENGTH"
echo "Batch Size (samples): $BATCH_SIZE_SAMPLES"
echo ""

# Check and change to work directory
if [ ! -d "$SAVE_DIR" ]; then
    echo "Error: Save directory does not exist: $SAVE_DIR"
    exit 1
fi
cd "$SAVE_DIR" || exit 1

echo "[Step 2/4] Creating directories..."
mkdir -p "$CACHE_DIR"
mkdir -p "$SAVE_DIR/MultiBitsQ/train_data/"
mkdir -p "$SAVE_DIR/MultiBitsQ/eval_data/"
mkdir -p "$SAVE_DIR/MultiBitsQ/model/"
echo "✓ Directories created"
echo ""

# Check if WORK_DIR/MultiBitsQ exists
if [ ! -d "$WORK_DIR/MultiBitsQ" ]; then
    echo "Error: $WORK_DIR/MultiBitsQ does not exist!"
    exit 1
fi

# Verify required files exist
MISSING_FILES=0
if [ ! -f "$INPUT_MODEL/config.json" ]; then
    echo "Warning: Input model not found at $INPUT_MODEL"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "Warning: Training data not found at $TRAIN_DATA_PATH"
    MISSING_FILES=$((MISSING_FILES + 1))
fi
if [ -n "$EVAL_DATA_PATH" ] && [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "Warning: Evaluation data not found at $EVAL_DATA_PATH"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo "Warning: $MISSING_FILES required file(s) are missing. The script will continue but may fail later."
    echo ""
fi

# Check and change to ParetoQ directory
if [ ! -d "$WORK_DIR/MultiBitsQ/ParetoQ" ]; then
    echo "Error: ParetoQ directory does not exist: $WORK_DIR/MultiBitsQ/ParetoQ"
    exit 1
fi
cd "$WORK_DIR/MultiBitsQ/ParetoQ" || exit 1

echo "[Step 3/4] Starting pretokenization..."
echo ""

if ! python pretokenize.py \
    --input_model_filename "$INPUT_MODEL" \
    --cache_dir "$CACHE_DIR" \
    --train_data_local_path "$TRAIN_DATA_PATH" \
    --eval_data_local_path "$EVAL_DATA_PATH" \
    --model_max_length $MODEL_MAX_LENGTH \
    --batch_size_samples $BATCH_SIZE_SAMPLES; then
    echo "Error: Pretokenization failed!"
    exit 1
fi

echo ""
echo "[Step 4/4] Pretokenization completed successfully!"
echo "========================================="
echo "Cache location: $CACHE_DIR"
echo ""
echo "To use the cached data in training, set:"
echo "  --cache_dir $CACHE_DIR"
echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="

