#!/bin/bash
# add noise injection and pre-quantization noise to the training
# Parent Directory of MultiBitQ，e.g. /fast/sliu/wanqi
source /fast/sliu/envs/multibitsq_env/bin/activate
WORK_DIR="/home/sliu/kwang/"

INPUT_MODEL="$WORK_DIR/LLM-Research/Llama-3.2-1B"
EXP_NAME="llama3_1b_234bit_12wsteps_bs128_lr2e-5_wonoise_asym"
BIT_LIST="2,3,4"
PROB_LIST="1,1,1"
EVAL_BITS_LIST="2,3,4"
MAX_STEPS=240000
BATCH_SIZE=8
ACCU_STEP=2
LEARNING_RATE=2e-5
DISABLE_CLIPVALS=True
CONTAIN_WEIGHT_CLIP_VAL=False
NOISE_INJECTION=False
PRE_QUANTIZATION_NOISE=False
RANDOM_ASSIGN=True
EVAL_BATCH_SIZE=8
OUTPUT_MODEL_FILENAME="llama3-1B-multibitq"

# Wandb configuration
export WANDB_API_KEY=799de0cca57925184d04f4c0b6588ff554c3b9ec
export WANDB_PROJECT="MultiBitsQ"
export WANDB_ENTITY="yangwq177-qti"
export WANDB_RUN_NAME="$EXP_NAME"

echo "========================================="
echo "Starting Multi-Bit Training Pipeline"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo ""

echo "[Step 1/6] Creating directories..."
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/log/
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/checkpoint/
echo "✓ Directories created"
echo ""

cd $WORK_DIR

echo "[Step 2/6] Installing requirements..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate multibitsq_env
pip install -r MultiBitsQ/ParetoQ/requirement.txt

echo "✓ Requirements installed"
echo ""

echo "[Step 3/6] Downloading data and models..."
# python MultiBitsQ/scripts/download_data.py
python MultiBitsQ/scripts/download_model.py
# python MultiBitsQ/scripts/download_wiki.py
echo "✓ Data and models downloaded"
echo ""

cd $WORK_DIR/MultiBitsQ/ParetoQ

echo "[Step 4/6] Starting training phase..."
echo "Training configuration:"
echo "  - INPUT_MODEL: $INPUT_MODEL"
echo "  - Multi-bit: $BIT_LIST"
echo "  - Probability: $PROB_LIST"
echo "  - Max steps: $MAX_STEPS"
echo "  - Batch size: $BATCH_SIZE (per device)"
echo "  - Random Assign: $RANDOM_ASSIGN"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Noise Injection: $NOISE_INJECTION"
echo "  - Pre-quantization Noise: $PRE_QUANTIZATION_NOISE"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
--output_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/checkpoint/" \
--input_model_filename "$INPUT_MODEL" \
--output_model_filename $OUTPUT_MODEL_FILENAME \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/log" \
--max_steps $MAX_STEPS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps $ACCU_STEP \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 20000 \
--report_to "wandb" \
--save_total_limit 12 \
--learning_rate $LEARNING_RATE \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits_list $BIT_LIST \
--prob_list $PROB_LIST \
--multiple_bits_random_assign $RANDOM_ASSIGN \
--multiple_bits_random_assign_prob 0.5 \
--multiple_bits_share_clipvals False \
--multiple_bits_disable_clipvals $DISABLE_CLIPVALS \
--noise_injection $NOISE_INJECTION \
--noise_sigma_weights 0.001 \
--noise_sigma_clipvals 0.001 \
--pre_quantization_noise $PRE_QUANTIZATION_NOISE \
--post_quantization_noise False 


OUTPUT_MODEL="$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/models/$OUTPUT_MODEL_FILENAME"
echo ""
echo "✓ Training completed"
echo "  - OUTPUT_MODEL: $OUTPUT_MODEL"
echo ""

echo "[Step 5/6] Starting evaluation phase..."


echo "Evaluation configuration:"
echo "  - Model: $OUTPUT_MODEL"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits_list: $BIT_LIST"
echo "  - eval_bit_list: $EVAL_BITS_LIST"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
--input_model_filename $OUTPUT_MODEL \
--output_model_filename "$OUTPUT_MODEL_FILENAME-1234bit_eval" \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/eval_log/" \
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
--multiple_bits_disable_clipvals $DISABLE_CLIPVALS 



echo ""
echo "✓ Evaluation completed"
echo ""

echo "[Step 6/6] Starting checkpoint evaluation phase..."
CHECKPOINT_DIR="$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/checkpoint/"

# Function to evaluate a checkpoint with different w_bits
evaluate_checkpoint() {
    local CHECKPOINT_PATH=$1
    local CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)
    
    echo ""
    echo "Evaluating checkpoint: $CHECKPOINT_NAME"
    echo "========================================="
    
    # Evaluate with w_bits=2
    echo "Evaluation configuration:"
    echo "  - Checkpoint: $CHECKPOINT_NAME"
    echo "  - Eval data: wikitext_10k_samples.jsonl"
    echo "  - w_bits_list: $BIT_LIST"
    echo "  - eval_bit_list: $EVAL_BITS_LIST"
    echo ""
    
    torchrun --nnodes=1 --nproc_per_node=8 train.py \
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
    --w_bits 2 \
    --w_bits_list $BIT_LIST \
    --eval_bit_list $EVAL_BITS_LIST \
    --contain_weight_clip_val $CONTAIN_WEIGHT_CLIP_VAL \
    --multiple_bits_disable_clipvals $DISABLE_CLIPVALS
    

    
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
        
        for CHECKPOINT in $CHECKPOINTS; do
            evaluate_checkpoint "$CHECKPOINT"
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