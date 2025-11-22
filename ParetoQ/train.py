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
from transformers import default_data_collator, Trainer, TrainerCallback

log = utils.get_logger("clm")


class MultiBitEvalCallback(TrainerCallback):
    """Callback to evaluate model on all bit widths in w_bits_list during training."""
    
    def __init__(self, w_bits_list, eval_steps, eval_dataset):
        self.w_bits_list = w_bits_list
        self.eval_steps = eval_steps
        self.eval_dataset = eval_dataset
        self.last_eval_step = -1
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate model on all bit widths at specified step intervals."""
        # Check if we should evaluate at this step
        if state.global_step % self.eval_steps != 0:
            return
        
        # Avoid duplicate evaluations at the same step
        if state.global_step == self.last_eval_step:
            return
        
        # Check if eval dataset is available
        if self.eval_dataset is None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                log.warning("No eval dataset available for multi-bit evaluation")
            return
        
        self.last_eval_step = state.global_step
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            log.info(f"Starting multi-bit evaluation at step {state.global_step}")
        
        # Collect all QuantizeLinear layers
        quant_layers = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizeLinear):
                quant_layers.append((name, module))
        
        if len(quant_layers) == 0:
            if is_main_process:
                log.warning("No QuantizeLinear layers found in model")
            return
        
        # Save original states for all layers
        original_states = {}
        for name, layer in quant_layers:
            original_states[name] = {
                'multiple_bits_random_assign': layer.multiple_bits_random_assign,
                'noise_injection': layer.noise_injection,
                'cur_w_bits': layer.cur_w_bits,
            }
        
        # Set model to eval mode
        model.eval()
        
        trainer = kwargs.get('trainer')
        if trainer is None:
            if is_main_process:
                log.error("Trainer not available in callback")
            return
        
        # Evaluate on each bit width
        for w_bits in self.w_bits_list:
            if is_main_process:
                log.info(f"Evaluating at {w_bits}-bit...")
            
            # Set bit width and disable random assignment and noise for all layers
            for name, layer in quant_layers:
                try:
                    layer.set_bits(w_bits)
                    layer.multiple_bits_random_assign = False
                    layer.noise_injection = False
                except ValueError as e:
                    if is_main_process:
                        log.warning(f"Failed to set {w_bits}-bit for layer {name}: {e}")
                    continue
            
            # Run evaluation
            try:
                with torch.no_grad():
                    metrics = trainer.evaluate(eval_dataset=self.eval_dataset)
                
                # Calculate perplexity
                eval_loss = metrics.get("eval_loss", float("inf"))
                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")
                
                # Log metrics with bit-specific names
                metrics_to_log = {
                    f"multibit_eval_loss_{w_bits}bit": eval_loss,
                    f"multibit_perplexity_{w_bits}bit": perplexity,
                }
                
                # Log to trainer's state and logging backends (tensorboard/wandb)
                # Only on main process to avoid duplicates
                if is_main_process:
                    # Add to trainer's log history
                    if hasattr(trainer.state, 'log_history'):
                        log_entry = {
                            'step': state.global_step,
                            **metrics_to_log
                        }
                        trainer.state.log_history.append(log_entry)
                    
                    # Log metrics using trainer's logging system
                    # trainer.log() automatically logs to all configured backends (tensorboard/wandb)
                    # based on the report_to parameter in TrainingArguments
                    # It uses trainer.state.global_step for the step number
                    trainer.log(metrics_to_log)
                    
                    # Also use log_metrics for explicit logging to ensure it's recorded
                    # This is similar to how regular eval metrics are logged (see line 364)
                    trainer.log_metrics("multibit_eval", metrics_to_log)
                    
                    log.info(f"Step {state.global_step} - {w_bits}-bit: eval_loss={eval_loss:.4f}, perplexity={perplexity:.4f}")
                
            except Exception as e:
                if is_main_process:
                    log.error(f"Error during evaluation at {w_bits}-bit: {e}")
                continue
        
        # Restore original states
        for name, layer in quant_layers:
            if name in original_states:
                orig_state = original_states[name]
                layer.multiple_bits_random_assign = orig_state['multiple_bits_random_assign']
                layer.noise_injection = orig_state['noise_injection']
                layer.cur_w_bits = orig_state['cur_w_bits']
        
        # Set model back to train mode
        model.train()
        
        if is_main_process:
            log.info(f"Completed multi-bit evaluation at step {state.global_step}")


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
    
    # Prepare callbacks list
    callbacks = []
    
    # Add multi-bit evaluation callback if enabled
    if (training_args.multibit_eval_steps is not None 
        and model_args.w_bits_list is not None 
        and len(model_args.w_bits_list) > 1
        and training_args.do_train):
        if valid_data is None:
            log.warning("multibit_eval_steps is set but no eval dataset is available. Multi-bit evaluation will be skipped.")
        else:
            multibit_callback = MultiBitEvalCallback(
                w_bits_list=model_args.w_bits_list,
                eval_steps=training_args.multibit_eval_steps,
                eval_dataset=valid_data,
            )
            callbacks.append(multibit_callback)
            log.info(f"Multi-bit evaluation enabled: evaluating every {training_args.multibit_eval_steps} steps on bit widths {model_args.w_bits_list}")
    
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
        callbacks=callbacks if callbacks else None,
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
