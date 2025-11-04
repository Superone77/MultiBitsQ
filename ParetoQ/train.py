# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
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
    
    # Handle w_bits_list for multi-bit training
    if model_args.w_bits_list is not None:
        # Parse comma-separated list
        w_bits_list = [int(x.strip()) for x in model_args.w_bits_list.split(',')]
        config.w_bits_list = w_bits_list
        config.w_bits = w_bits_list[0]  # Set default to first bit
    else:
        config.w_bits = model_args.w_bits
        config.w_bits_list = None
    
    # Set noise injection parameters
    config.noise_injection = model_args.noise_injection
    config.noise_sigma_weights = model_args.noise_sigma_weights
    config.noise_sigma_clipvals = model_args.noise_sigma_clipvals
    config.pre_quantization_noise = model_args.pre_quantization_noise
    config.post_quantization_noise = model_args.post_quantization_noise
    config.multiple_bits_random_assign = model_args.multiple_bits_random_assign
    config.multiple_bits_random_assign_prob = model_args.multiple_bits_random_assign_prob
    config.multiple_bits_share_clipvals = model_args.multiple_bits_share_clipvals
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    if not model_args.contain_weight_clip_val:
        # Determine which bits to initialize
        if model_args.w_bits_list is not None:
            w_bits_to_init = [int(x.strip()) for x in model_args.w_bits_list.split(',')]
        else:
            w_bits_to_init = [model_args.w_bits]
        
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)
                
                if weight_param is None:
                    continue
                
                # Check if weight_param contains NaN or Inf
                if torch.isnan(weight_param).any() or torch.isinf(weight_param).any():
                    log.error(f"[train] weight_param contains NaN/Inf for {weight_name}!")
                    nan_count = torch.isnan(weight_param).sum().item() if torch.isnan(weight_param).any() else 0
                    inf_count = torch.isinf(weight_param).sum().item() if torch.isinf(weight_param).any() else 0
                    log.error(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                    # Replace NaN/Inf with zeros for safety
                    weight_param = torch.where(torch.isnan(weight_param) | torch.isinf(weight_param), 
                                               torch.zeros_like(weight_param), 
                                               weight_param)
                
                # Determine which bit width this clip_val corresponds to
                # Check if it's from a list (format: "weight_clip_val_list.2", "weight_clip_val_list.3", etc.)
                w_bits = None
                if "weight_clip_val_list" in name:
                    # Extract bit width from parameter name, e.g., "weight_clip_val_list.2"
                    try:
                        bit_str = name.split('.')[-1]
                        w_bits = int(bit_str)
                    except (ValueError, IndexError):
                        # If can't parse, use first bit in list
                        w_bits = w_bits_to_init[0] if w_bits_to_init else model_args.w_bits
                else:
                    # Single clip_val, use first bit or w_bits
                    w_bits = w_bits_to_init[0] if w_bits_to_init else model_args.w_bits

                if w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
                elif w_bits == 0 or w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    # Check for NaN/Inf in scale
                    if torch.isnan(scale).any() or torch.isinf(scale).any():
                        log.error(f"[train] NaN/Inf detected in scale calculation for {name}! w_bits={w_bits}")
                        log.error(f"  weight_param stats: min={weight_param.min().item():.6f}, max={weight_param.max().item():.6f}")
                        # Replace NaN/Inf with a safe default
                        scale = torch.where(torch.isnan(scale) | torch.isinf(scale),
                                           torch.ones_like(scale) * 0.1,  # Default safe value
                                           scale)
                    # Debug: Check for very small scale values
                    if w_bits == 2:
                        min_scale = scale.min().item()
                        if min_scale < 0.01:
                            log.warning(f"[train] w_bits=2: Found very small scale value: {min_scale:.6f} for {name}")
                            log.warning(f"  weight_param stats: min={weight_param.min().item():.6f}, max={weight_param.max().item():.6f}")
                    # Add minimum protection for w_bits=2
                    if w_bits == 2:
                        scale = torch.clamp(scale, min=0.01)
                elif w_bits == 3 or w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (w_bits - 1) - 1
                    scale = xmax / maxq
                    # Check for NaN/Inf
                    if torch.isnan(scale).any() or torch.isinf(scale).any():
                        log.error(f"[train] NaN/Inf detected in scale for {name}! w_bits={w_bits}")
                        scale = torch.where(torch.isnan(scale) | torch.isinf(scale),
                                           torch.ones_like(scale) * 0.1,
                                           scale)
                else:
                    # For > 4 bits, use a default initialization
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (w_bits - 1) - 1
                    scale = xmax / maxq if maxq > 0 else xmax
                    # Check for NaN/Inf
                    if torch.isnan(scale).any() or torch.isinf(scale).any():
                        log.error(f"[train] NaN/Inf detected in scale for {name}! w_bits={w_bits}")
                        scale = torch.where(torch.isnan(scale) | torch.isinf(scale),
                                           torch.ones_like(scale) * 0.1,
                                           scale)

                # Final check: Ensure scale is valid before copying
                if torch.isnan(scale).any() or torch.isinf(scale).any():
                    log.error(f"[train] NaN/Inf still present in scale for {name} after all checks! w_bits={w_bits}")
                    log.error(f"  scale stats: min={scale.min().item():.6f}, max={scale.max().item():.6f}")
                    # Force replace with safe values
                    scale = torch.where(torch.isnan(scale) | torch.isinf(scale),
                                       torch.ones_like(scale) * 0.1,
                                       scale)
                    scale = torch.clamp(scale, min=0.01, max=10.0)  # Reasonable bounds

                param.data.copy_(scale)
                
                # Final verification after copy
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    log.error(f"[train] NaN/Inf detected in param.data after copy for {name}! w_bits={w_bits}")
                    # Emergency fix: replace NaN/Inf with safe values
                    param.data.copy_(torch.where(torch.isnan(param.data) | torch.isinf(param.data),
                                                 torch.ones_like(param.data) * 0.1,
                                                 param.data))
                    param.data.clamp_(min=0.01, max=10.0)
                    log.warning(f"[train] Fixed param.data by replacing NaN/Inf with safe values")
                
                if w_bits == 2:
                    min_param = param.data.min().item()
                    max_param = param.data.max().item()
                    if min_param < 0.01:
                        log.warning(f"[train] w_bits=2: After copy, param.data min={min_param:.6f} for {name}")
                    if max_param > 10.0:
                        log.warning(f"[train] w_bits=2: After copy, param.data max={max_param:.6f} for {name}")

    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
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
