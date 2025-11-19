#!/bin/bash

source /fast/sliu/envs/multibitsq_env/bin/activate

# Parent Directory of MultiBitQ，e.g. /fast/sliu/wanqi
WORK_DIR="/home/sliu/kwang/"


echo "========================================="
echo "Starting Multi-Bit Training Pipeline"
echo "========================================="
echo "Working Directory: $WORK_DIR"
echo ""

echo "[Step 1/6] Creating directories..."
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm/
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/log/
mkdir -p $WORK_DIR/MultiBitsQ/ParetoQ/tmp/checkpoint/
echo "✓ Directories created"
echo ""

cd $WORK_DIR

echo "[Step 2/6] Installing requirements..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate multibitsq_env
pip install -r MultiBitsQ/ParetoQ/requirements.txt

echo "✓ Requirements installed"
echo ""

echo "[Step 3/6] Downloading data and models..."
python MultiBitsQ/scripts/download_data.py
python MultiBitsQ/scripts/download_model.py
python MultiBitsQ/scripts/download_wiki.py
echo "✓ Data and models downloaded"
echo ""

cd $WORK_DIR/MultiBitsQ/ParetoQ

echo "[Step 4/6] Starting training phase..."
echo "Training configuration:"
echo "  - Model: MobileLLM-125M"
echo "  - Multi-bit: 4,3,2,1"
echo "  - Max steps: 120000"
echo "  - Batch size: 16 (per device)"
echo "  - Learning rate: 2e-5"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/" \
--output_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/checkpoint/" \
--input_model_filename "$WORK_DIR/LLM-Research/MobileLLM-125M" \
--output_model_filename "mobilellm-1234bit" \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/log" \
--max_steps 120000 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5000 \
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
--w_bits_list "4,3,2" \
--multiple_bits_random_assign True \
--multiple_bits_random_assign_prob 0.5 \
--multiple_bits_share_clipvals False \
--multiple_bits_disable_clipvals False \
--noise_injection False \
--noise_sigma_weights 0.001 \
--noise_sigma_clipvals 0.001 \
--pre_quantization_noise False \
--post_quantization_noise False 

echo ""
echo "✓ Training completed"
echo ""

echo "[Step 5/6] Starting evaluation phase..."
# echo "Evaluation configuration:"
# echo "  - Model: mobilellm-1234bit (trained model)"
# echo "  - Eval data: wikitext_10k_samples.jsonl"
# echo "  - w_bits: 1"
# echo "  - w_bits_list: 1,2,3,4"
# echo ""

# torchrun --nnodes=1 --nproc_per_node=8 train.py \
# --local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/" \
# --input_model_filename "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/models/mobilellm-1234bit" \
# --output_model_filename "mobilellm-1234bit_eval" \
# --train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
# --eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
# --do_train False \
# --do_eval True \
# --model_max_length 2048 \
# --fp16 False \
# --bf16 True \
# --log_on_each_node False \
# --logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/eval_log/" \
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
echo "  - Model: mobilellm-1234bit (trained model)"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 2"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/" \
--input_model_filename "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/models/mobilellm-1234bit" \
--output_model_filename "mobilellm-1234bit_eval" \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/eval_log/" \
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
echo "  - Model: mobilellm-1234bit (trained model)"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 3"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/" \
--input_model_filename "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/models/mobilellm-1234bit" \
--output_model_filename "mobilellm-1234bit_eval" \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/eval_log/" \
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
echo "  - Model: mobilellm-1234bit (trained model)"
echo "  - Eval data: wikitext_10k_samples.jsonl"
echo "  - w_bits: 4"
echo "  - w_bits_list: 1,2,3,4"
echo ""

torchrun --nnodes=1 --nproc_per_node=8 train.py \
--local_dir "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/" \
--input_model_filename "$WORK_DIR/MultiBitsQ/ParetoQ/tmp/mobilellm_125M_1234bit_12wsteps_bs128_lr2e-5_noise_w0001_c0001/models/mobilellm-1234bit" \
--output_model_filename "mobilellm-1234bit_eval" \
--train_data_local_path "$WORK_DIR/finewebedu_50k_samples.jsonl" \
--eval_data_local_path "$WORK_DIR/wikitext_10k_samples.jsonl" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "$WORK_DIR/MultiBitsQ/ParetoQ/eval_log/" \
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