# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.utils_quant import QuantizeLinear
import copy
import torch
import transformers
from utils import utils
from utils import datautils

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer

log = utils.get_logger("clm")


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    config = LlamaConfig.from_pretrained(model_args.input_model_filename)
    config.w_bits = model_args.w_bits
    # Set multi-bit and noise injection parameters
    if model_args.w_bits_list is not None:
        config.w_bits_list = model_args.w_bits_list
    if model_args.prob_list is not None:
        config.prob_list = model_args.prob_list
    config.noise_injection = model_args.noise_injection
    config.noise_sigma_weights = model_args.noise_sigma_weights
    config.noise_sigma_clipvals = model_args.noise_sigma_clipvals
    config.initialize_noise = model_args.initialize_noise
    config.pre_quantization_noise = model_args.pre_quantization_noise
    config.post_quantization_noise = model_args.post_quantization_noise
    config.trainable_noise_scale = model_args.trainable_noise_scale
    config.multiple_bits_random_assign = model_args.multiple_bits_random_assign
    config.multiple_bits_random_assign_prob = model_args.multiple_bits_random_assign_prob
    config.multiple_bits_share_clipvals = model_args.multiple_bits_share_clipvals
    config.multiple_bits_disable_clipvals = model_args.multiple_bits_disable_clipvals
    config.use_stretch = model_args.use_stretch
    config.stretch_alpha = model_args.stretch_alpha
    
    # Set MobileLLM specific parameters
    config.share_embedding = model_args.share_embedding
    config.layer_sharing = model_args.layer_sharing
    
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    if not model_args.contain_weight_clip_val:
        # Determine which bit widths to initialize
        w_bits_to_init = []
        if model_args.w_bits_list is not None:
            w_bits_to_init = model_args.w_bits_list
        else:
            w_bits_to_init = [model_args.w_bits]
        
        named_params = dict(model.named_parameters())
        for name, param in named_params.items():
            if "weight_clip_val" in name:
                if "weight_clip_val_list" in name:
                    # name example: module.weight_clip_val_list.4
                    weight_prefix, _, bit_suffix = name.partition(".weight_clip_val_list.")
                    weight_name = f"{weight_prefix}.weight"
                else:
                    weight_name = name.replace("weight_clip_val", "weight")
                    bit_suffix = None
                weight_param = named_params.get(weight_name, None)
                
                if weight_param is None:
                    continue
                
                # Extract bit width from parameter name if it's in a list format
                w_bits = model_args.w_bits  # Default
                if bit_suffix is not None:
                    # Extract bit width from name like "model.layers.0.self_attn.q_proj.weight_clip_val_list.1"
                    # Find the number after the last dot
                    if bit_suffix:
                        try:
                            w_bits = int(bit_suffix.split(".")[-1])
                        except ValueError:
                            w_bits = model_args.w_bits
                else:
                    # For single bit width or shared clipvals, use the first bit width
                    if len(w_bits_to_init) > 0:
                        w_bits = w_bits_to_init[0]
                
                if w_bits >= 16:
                    continue
                elif w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
                elif w_bits == 0 or w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif w_bits == 3 or w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    # For higher bit widths, use similar logic
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (w_bits - 1) - 1
                    scale = xmax / maxq if maxq > 0 else xmax

                param.data.copy_(scale)

    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    # Use AutoTokenizer for better compatibility with different model types (including MobileLLM)
    # MobileLLM models may not support Fast tokenizer, so we check model name first
    model_name_lower = model_args.input_model_filename.lower()
    use_fast_tokenizer = not ("mobilellm" in model_name_lower or "mobile_llm" in model_name_lower)
    
    if use_fast_tokenizer:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_args.input_model_filename,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
            )
            log.info("Loaded fast tokenizer successfully.")
        except Exception as e:
            log.warning(f"Failed to load fast tokenizer: {e}. Falling back to slow tokenizer.")
            use_fast_tokenizer = False
    
    if not use_fast_tokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model_filename,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,  # Use slow tokenizer for MobileLLM or if fast tokenizer failed
        )
        log.info("Loaded slow tokenizer.")
    
    # Set tokenizer defaults
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set tokenizer defaults for backward compatibility
    if hasattr(tokenizer, 'add_bos_token'):
        tokenizer.add_bos_token = False
    if hasattr(tokenizer, 'add_eos_token'):
        tokenizer.add_eos_token = False
    log.info("Complete tokenizer loading...")

    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    model.config.use_cache = False
    myTrainer = Trainer
    
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
