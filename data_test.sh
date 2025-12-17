#!/bin/bash
set -euo pipefail

# Defaults mirror scripts/0_llama3_1b_new.sh but can be overridden via env.
MODEL_PATH="${MODEL_PATH:-/fast/wangk/MultiBitsQ/model/LLM-Research/Llama-3___2-1B}"
TRAIN_DATA="${TRAIN_DATA:-/fast/wangk/MultiBitsQ/train_data/finewebedu_train_samples.jsonl}"
EVAL_DATA="${EVAL_DATA:-}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_SEQS="${MAX_SEQS:-0}"
PRINT_INTERVAL="${PRINT_INTERVAL:-100000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/ParetoQ"

MAX_SEQ_ARGS=()
if [ "${MAX_SEQS}" -gt 0 ]; then
  MAX_SEQ_ARGS=(--max_sequences "${MAX_SEQS}")
fi

echo "Checking train dataset tokens against vocab..."
python datautils_test.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${TRAIN_DATA}" \
  --dataset_label "train" \
  --block_size "${BLOCK_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --print_interval "${PRINT_INTERVAL}" \
  "${MAX_SEQ_ARGS[@]}"

if [ -n "${EVAL_DATA}" ]; then
  echo "Checking eval dataset tokens against vocab..."
  python datautils_test.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${EVAL_DATA}" \
    --dataset_label "eval" \
    --block_size "${BLOCK_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --print_interval "${PRINT_INTERVAL}" \
    "${MAX_SEQ_ARGS[@]}"
fi
