# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=3 
torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "/local/mnt/workspace/wanqi/tmp/MultiBitsQ/ParetoQ/tmp/mobilellm_test/" \
--input_model_filename "/local/mnt/workspace/wanqi/tmp/MultiBitsQ/ParetoQ/tmp/mobilellm_debug/models/mobilellm" \
--output_model_filename "1B-finetuned" \
--train_data_local_path "/local/mnt/workspace/wanqi/tmp/MultiBitsQ/finewebedu_10k_samples.jsonl" \
--eval_data_local_path "/local/mnt/workspace/wanqi/tmp/MultiBitsQ/wikitext_10k_samples.jsonl" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
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
--w_bits_list '2,3,4' \
--contain_weight_clip_val True 
