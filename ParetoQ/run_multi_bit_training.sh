#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Multi-bit training script with noise injection support
# This script enables training with multiple bit widths and noise injection
#
# Usage examples:
#   Option 1: Multi-bit training with noise injection (recommended)
#     --w_bits_list "2,3,4" --noise_injection True --pre_quantization_noise True
#
#   Option 2: Single bit training (backward compatible)
#     --w_bits 4 --noise_injection False
#
#   Option 3: Multi-bit without noise
#     --w_bits_list "2,3,4" --noise_injection False

torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "meta-llama/Llama-3.2-1B" \
--output_model_filename "1B-multibit-trained" \
--train_data_local_path "/tmp/train.jsonl" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/multibit \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
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
--w_bits_list "2,3,4" \
--noise_injection False \
--noise_sigma_weights 0.001 \
--noise_sigma_clipvals 0.001 \
--pre_quantization_noise False \
--post_quantization_noise False \
--multiple_bits_random_assign False \
--multiple_bits_random_assign_prob 0.5 \
--multiple_bits_share_clipvals False
