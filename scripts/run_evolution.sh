#!/bin/bash
# Script to run evolutionary search for mixed-precision quantization
# 
# This script runs the evolution algorithm to find optimal bit-width assignments
# for different layers in a quantized model.
#
# Usage: ./run_evolution.sh
#   - Modify the variables below to customize the evolution parameters
#   - Ensure model_path and eval_data point to valid files

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration Parameters
# ============================================================================

# Model and data paths
MODEL_PATH="/path/to/supernet"
EVAL_DATA="/path/to/val.jsonl"
SAVE_RESULT_PATH="/tmp/best_bits.json"

# Evolution algorithm parameters
CANDIDATE_BITS="2,3,4"          # Comma-separated candidate bit widths
POPULATION_SIZE=12              # Number of individuals in population
GENERATIONS=8                   # Number of evolution generations
TOP_K=4                         # Number of top individuals kept each generation
MUTATION_PROB=0.15              # Probability of mutation
CROSSOVER_PROB=0.6              # Probability of crossover

# Constraints
MAX_AVG_BITS=3.0                # Maximum average bit-width constraint
# MAX_BIT_BUDGET=                # Optional: total bit budget (uncomment to use)

# Evaluation parameters
EVAL_BATCH_SIZE=4               # Batch size for evaluation
MAX_EVAL_SAMPLES=512            # Maximum number of evaluation samples
BLOCK_SIZE=512                  # Block size for tokenization
# NUM_PROC=                      # Optional: parallel workers for tokenization

# Other parameters
DTYPE="bf16"                    # Computation dtype: fp16, bf16, or fp32
SEED=42                         # Random seed for reproducibility

# ============================================================================
# Environment Setup (Optional)
# ============================================================================

# Uncomment and modify if you need to activate a virtual environment
# if [ -f "/path/to/your/venv/bin/activate" ]; then
#     source /path/to/your/venv/bin/activate
# fi

# ============================================================================
# Validation
# ============================================================================

if [ ! -f "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Evaluation data file not found: $EVAL_DATA"
    exit 1
fi

# ============================================================================
# Run Evolution Algorithm
# ============================================================================

echo "========================================="
echo "Evolutionary Search for Mixed-Precision Quantization"
echo "========================================="
echo "Model Path: $MODEL_PATH"
echo "Eval Data: $EVAL_DATA"
echo "Candidate Bits: $CANDIDATE_BITS"
echo "Population Size: $POPULATION_SIZE"
echo "Generations: $GENERATIONS"
echo "Max Avg Bits: $MAX_AVG_BITS"
echo "========================================="
echo ""

python ParetoQ/evolution.py \
    --model_path "$MODEL_PATH" \
    --eval_data "$EVAL_DATA" \
    --candidate_bits "$CANDIDATE_BITS" \
    --population_size "$POPULATION_SIZE" \
    --generations "$GENERATIONS" \
    --top_k "$TOP_K" \
    --mutation_prob "$MUTATION_PROB" \
    --crossover_prob "$CROSSOVER_PROB" \
    --max_avg_bits "$MAX_AVG_BITS" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --block_size "$BLOCK_SIZE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --save_result_path "$SAVE_RESULT_PATH"

echo ""
echo "========================================="
echo "Evolution completed!"
echo "Results saved to: $SAVE_RESULT_PATH"
echo "========================================="