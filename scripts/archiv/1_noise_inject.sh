#!/bin/bash
# add noise injection and pre-quantization noise to the training
# Parent Directory of MultiBitQ，e.g. /fast/sliu/wanqi
source /fast/sliu/envs/multibitsq_env/bin/activate
WORK_DIR="/home/sliu/kwang/"

INPUT_MODEL="$WORK_DIR/LLM-Research/Llama-3.2-1B"
EXP_NAME="llama3-1B_234bit_12wsteps_bs128_lr2e-5_w_noise_0001_0001"
BIT_LIST="2,3,4"
MAX_STEPS=240000
BATCH_SIZE=8
ACCU_STEP=2
LEARNING_RATE=2e-5
NOISE_INJECTION=True
PRE_QUANTIZATION_NOISE=True
RANDOM_ASSIGN=True


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
# python MultiBitsQ/scripts/download_model.py
# python MultiBitsQ/scripts/download_wiki.py
echo "✓ Data and models downloaded"
echo ""

cd $WORK_DIR/MultiBitsQ/ParetoQ

echo "[Step 4/6] Starting training phase..."
echo "Training configuration:"
echo "  - INPUT_MODEL: $INPUT_MODEL"
echo "  - Multi-bit: $BIT_LIST"
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
--output_model_filename "llama3-1B-multibitq" \
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
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps $ACCU_STEP \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate $LEARNING_RATE \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits_list $BIT_LIST \
--multiple_bits_random_assign $RANDOM_ASSIGN \
--multiple_bits_random_assign_prob 0.5 \
--multiple_bits_share_clipvals False \
--multiple_bits_disable_clipvals False \
--noise_injection $NOISE_INJECTION \
--noise_sigma_weights 0.001 \
--noise_sigma_clipvals 0.001 \
--pre_quantization_noise $PRE_QUANTIZATION_NOISE \
--post_quantization_noise False 


OUTPUT_MODEL="$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/models/llama3-1B-multibitq"
echo ""
echo "✓ Training completed"
echo "  - OUTPUT_MODEL: $OUTPUT_MODEL"
echo ""

echo "[Step 5/6] Starting evaluation phase..."
# echo "Evaluation configuration:"
# echo "  - Model: $OUTPUT_MODEL"
# echo "  - Eval data: wikitext_10k_samples.jsonl"
# echo "  - w_bits: 1"
# echo "  - w_bits_list: 1,2,3,4"
# echo ""

# torchrun --nnodes=1 --nproc_per_node=8 train.py \
# --local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
# --input_model_filename $OUTPUT_MODEL \
# --output_model_filename "llama3-1B-1234bit_eval" \
# --train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
# --eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
# --do_train False \
# --do_eval True \
# --model_max_length 2048 \
# --fp16 False \
# --bf16 True \
# --log_on_each_node False \
# --logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/$EXP_NAME/eval_log/" \
# --num_train_epochs 1 \
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 1 \
# --evaluation_strategy "no" \
# --save_strategy "steps" \
# --save_steps 2000 \
# --report_to "tensorboard" \
# --save_total_limit 1 \
# --learning_rate 2e-5 \
# --weight_decay 0. \
# --warmup_ratio 0. \
# --lr_scheduler_type "cosine" \
# --logging_steps 1 \
# --tf32 False \
# --gradient_checkpointing False \
# --qat True \
# --w_bits 1 \
# --w_bits_list '1,4' \
# --contain_weight_clip_val True 

echo "Evaluation configuration:"
echo "  - Model: $OUTPUT_MODEL"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 2"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
--input_model_filename $OUTPUT_MODEL \
--output_model_filename "llama3-1B-1234bit_eval" \
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
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
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
--w_bits_list '2,4' \
--contain_weight_clip_val True 

echo "Evaluation configuration:"
echo "  - Model: $OUTPUT_MODEL"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 3"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
--input_model_filename $OUTPUT_MODEL \
--output_model_filename "llama3-1B-1234bit_eval" \
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
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits 3 \
--w_bits_list '3,4' \
--contain_weight_clip_val True 

echo "Evaluation configuration:"
echo "  - Model: $OUTPUT_MODEL"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 4"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/$EXP_NAME/" \
--input_model_filename $OUTPUT_MODEL \
--output_model_filename "llama3-1B-1234bit_eval" \
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
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits 4 \
--w_bits_list '4,3' \
--contain_weight_clip_val True 

echo ""
echo "✓ Evaluation completed"
echo ""

echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="