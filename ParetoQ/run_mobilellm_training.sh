#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for training MobileLLM models with ParetoQ quantization
# This script demonstrates how to load a MobileLLM model from HuggingFace
# and train it with multi-bit quantization

# Example: MobileLLM-125M from HuggingFace
# Replace with your desired MobileLLM model, e.g.:
# - facebook/MobileLLM-125M
# - facebook/MobileLLM-350M
# - facebook/MobileLLM-600M
# - facebook/MobileLLM-1B
# - facebook/MobileLLM-1.5B

torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "facebook/MobileLLM-125M" \
--output_model_filename "MobileLLM-125M-multi-bit-trained" \
--train_data_local_path "/tmp/train.jsonl" \
--eval_data_local_path "/tmp/eval.jsonl" \
--do_train True \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 500 \
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
--w_bits_list "1,2,4" \
--w_bits 1 \
--multiple_bits_random_assign True \
--multiple_bits_random_assign_prob 0.5 \
--multiple_bits_share_clipvals False \
--multiple_bits_disable_clipvals False \
--noise_injection True \
--noise_sigma_weights 0.001 \
--noise_sigma_clipvals 0.001 \
--initialize_noise False \
--pre_quantization_noise True \
--post_quantization_noise False \
--trainable_noise_scale False \
--use_stretch False \
--stretch_alpha 1.0 \
--share_embedding True \
--layer_sharing False

