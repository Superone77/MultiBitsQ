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



try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except ImportError:
    simple_evaluate = None
    HFLM = None


def _flatten_lm_eval_results(results_dict, bit_suffix=None):
    """
    Convert lm_eval simple_evaluate output into a flat metrics dict so Trainer can log/save it.
    """
    metrics = {}
    suffix = f"_{bit_suffix}bit" if bit_suffix is not None else ""
    if results_dict is None:
        return metrics
    task_results = results_dict.get("results", {})
    for task, task_metrics in task_results.items():
        for metric_name, value in task_metrics.items():
            metrics[f"lm_eval_{task}_{metric_name}{suffix}"] = value
    return metrics


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    config = LlamaConfig.from_pretrained(model_args.input_model_filename)
    # Set multi-bit parameters
    if model_args.w_bits_list is not None:
        config.w_bits_list = model_args.w_bits_list
    else:
        raise ValueError("w_bits_list must be provided. For single-bit training, use w_bits_list with one element, e.g., '--w_bits_list 2'")
    if model_args.prob_list is not None:
        config.prob_list = model_args.prob_list
    config.multiple_bits_random_assign = model_args.multiple_bits_random_assign
    config.multiple_bits_random_assign_prob = model_args.multiple_bits_random_assign_prob
    
    
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    if not model_args.contain_weight_clip_val:
        # Determine which bit widths to initialize
        w_bits_to_init = model_args.w_bits_list
        
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
                w_bits = w_bits_to_init[0]  # Default to first bit width
                if bit_suffix is not None:
                    # Extract bit width from name like "model.layers.0.self_attn.q_proj.weight_clip_val_list.1"
                    # Find the number after the last dot
                    if bit_suffix:
                        try:
                            w_bits = int(bit_suffix.split(".")[-1])
                        except ValueError:
                            log.warning(f"Failed to extract bit width from parameter name: {name}")
                            w_bits = w_bits_to_init[0]
                else:
                    # For shared clipvals (weight_clip_val without list), use the first bit width
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
                    raise NotImplementedError
                    # xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    # maxq = 2 ** (w_bits - 1) - 1
                    # scale = xmax / maxq if maxq > 0 else xmax

                param.data.copy_(scale)

    model.cuda()
    log.info("Complete model loading...")
    
    # Set gradient_accumulation_steps for all QuantizeLinear layers
    gradient_accumulation_steps = getattr(training_args, 'gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        log.info(f"Setting gradient_accumulation_steps={gradient_accumulation_steps} for all QuantizeLinear layers")
        for name, module in model.named_modules():
            if isinstance(module, QuantizeLinear):
                module.gradient_accumulation_steps = gradient_accumulation_steps
    
    # Enable debug mode for specified layers
    if model_args.debug_layers is not None and len(model_args.debug_layers) > 0:
        log.info(f"Enabling debug mode for layers: {model_args.debug_layers}")
        debug_layers_set = set(model_args.debug_layers)
        found_layers = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizeLinear):
                # Check if this layer name matches any debug layer pattern
                # Support both exact match and partial match (e.g., "layers.0" matches "model.layers.0.self_attn.q_proj")
                should_debug = False
                for debug_layer in debug_layers_set:
                    if name == debug_layer or name.endswith('.' + debug_layer) or debug_layer in name:
                        should_debug = True
                        break
                
                if should_debug:
                    module.debug = True
                    module.layer_name = name
                    # Initialize logger for debug mode
                    import logging
                    module.logger = logging.getLogger("clm")
                    found_layers.append(name)
                    log.info(f"Enabled debug for layer: {name}")
        
        if len(found_layers) == 0:
            log.warning(f"No matching QuantizeLinear layers found for debug_layers: {model_args.debug_layers}")
        else:
            log.info(f"Total {len(found_layers)} layers enabled for debug mode")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
    log.info("Complete tokenizer loading...")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 训练集加载
    log.info("Using streaming tokenization for training data")
    train_data = datautils.StreamingJsonlDataset(
        path=data_args.train_data_local_path,
        tokenizer=tokenizer,
        block_size=training_args.model_max_length,
        rank=rank,
        world_size=world_size,
        max_samples=data_args.max_train_samples,
        batch_size=10000,  # [建议新增] 增大批处理大小以加速分词，默认是1000
    )

    valid_data = None
    if data_args.eval_data_local_path is not None:
        log.info("Using streaming tokenization for eval data")
        valid_data = datautils.StreamingJsonlDataset(
            path=data_args.eval_data_local_path,
            tokenizer=tokenizer,
            block_size=min(training_args.model_max_length, 1024),
            rank=rank,
            world_size=world_size,
            max_samples=data_args.max_eval_samples,
            batch_size=1000, # 验证集数据量通常较小，默认即可
    )        
    
    eval_data_len = datautils.get_dataset_length(valid_data) if valid_data is not None else None
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

    # Periodically log quantized losses (2/3/4 bit) to wandb on logging_steps
    quant_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, QuantizeLinear)]
    quant_log_bits = model_args.w_bits_list
    if training_args.do_train and training_args.logging_steps > 0 and len(quant_layers) > 0:
        quant_eval_dataset = valid_data if valid_data is not None else train_data
        if quant_eval_dataset is None:
            log.warning("Quantized loss logging is enabled but no dataset is available; skipping periodic quantized logging.")
        else:
            # Limit evaluation overhead to a small number of batches
            max_eval_batches = 1
            if (
                hasattr(data_args, "max_eval_samples")
                and data_args.max_eval_samples is not None
                and data_args.max_eval_samples > 0
                and training_args.per_device_eval_batch_size > 0
            ):
                max_eval_batches = max(1, math.ceil(data_args.max_eval_samples / training_args.per_device_eval_batch_size))

            class QuantizedLossLoggingCallback(TrainerCallback):
                def __init__(self, trainer_ref):
                    self.trainer = trainer_ref
                    self.quant_layers = quant_layers
                    self.quant_bits = quant_log_bits
                    self.eval_dataset = quant_eval_dataset
                    self.max_eval_batches = max_eval_batches
                    self.in_progress = False

                def on_log(self, args, state, control, **kwargs):
                    # Trigger quantized logging right after the default Trainer log fires
                    if self.in_progress or state.global_step == 0:
                        return control

                    # Run quant logging every other logging event so it is separated from the default log line
                    if state.global_step % (args.logging_steps * 2) != 0:
                        return control

                    # Keep all ranks in sync before starting evaluation
                    if dist.is_initialized():
                        dist.barrier()

                    self.in_progress = True
                    try:
                        metrics = self._compute_quantized_losses(kwargs.get("model", self.trainer.model))
                    finally:
                        self.in_progress = False
                        if dist.is_initialized():
                            dist.barrier()

                    # Only the logging process reports to wandb/console
                    if metrics and args.should_log:
                        if dist.get_rank() == 0:
                            log.info(metrics)
                    return control

                def _compute_quantized_losses(self, model):
                    if model is None or self.eval_dataset is None:
                        return {}

                    # Unwrap DistributedDataParallel if needed
                    model_to_use = model.module if hasattr(model, "module") else model

                    original_states = {}
                    for name, layer in self.quant_layers:
                        original_states[name] = {
                            "multiple_bits_random_assign": getattr(layer, "multiple_bits_random_assign", None),
                            "cur_w_bits": getattr(layer, "cur_w_bits", None),
                        }

                    losses = {}
                    for w_bits in self.quant_bits:
                        available = False
                        for name, layer in self.quant_layers:
                            if w_bits not in getattr(layer, "w_bits_list", []):
                                continue
                            available = True
                            try:
                                layer.set_bits(w_bits)
                                if hasattr(layer, "multiple_bits_random_assign"):
                                    layer.multiple_bits_random_assign = False
                            except ValueError as exc:
                                if self.trainer.args.should_log:
                                    log.warning(f"Failed to set {w_bits}-bit for layer {name}: {exc}")
                                continue

                        if not available:
                            continue

                        loss_val = self._evaluate_loss(model, w_bits)
                        if loss_val is not None:
                            losses[f"quant_loss_{w_bits}bit"] = loss_val

                    # Restore original bit/random assignment state
                    for name, layer in self.quant_layers:
                        state = original_states.get(name, {})
                        if state.get("multiple_bits_random_assign") is not None:
                            layer.multiple_bits_random_assign = state["multiple_bits_random_assign"]
                        if state.get("cur_w_bits") is not None:
                            layer.cur_w_bits = state["cur_w_bits"]

                    model_to_use.train()
                    return losses

                def _evaluate_loss(self, model, w_bits):
                    dataloader = self.trainer.get_eval_dataloader(self.eval_dataset)
                    device = self.trainer.args.device
                    losses = []
                    model.eval()
                    with torch.no_grad():
                        for step, batch in enumerate(dataloader):
                            if step >= self.max_eval_batches:
                                break
                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = getattr(outputs, "loss", None)
                            if loss is None:
                                continue
                            losses.append(loss.detach())

                    if not losses:
                        return None

                    loss_tensor = torch.stack(losses).mean()
                    if dist.is_initialized():
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                    return loss_tensor.item()

            trainer.add_callback(QuantizedLossLoggingCallback(trainer))

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)
        log.info(f"Rank {rank} token counts: {trainer.get_token_counts()}")

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        has_eval_bit_list = model_args.eval_bit_list is not None and len(model_args.eval_bit_list) > 0
        has_eval_dataset = valid_data is not None

        def _sample_count_from_metrics(metrics: dict | None):
            if metrics is None:
                return None
            sample_count = metrics.get("eval_samples")
            if sample_count is not None:
                return sample_count

            dataset_len = datautils.get_dataset_length(valid_data) if has_eval_dataset else None
            eval_max_samples = getattr(data_args, "max_eval_samples", None)
            if eval_max_samples is not None and dataset_len is not None:
                return min(eval_max_samples, dataset_len)
            if eval_max_samples is not None:
                return eval_max_samples
            return dataset_len

        quant_layers = []
        if has_eval_bit_list:
            for name, module in model.named_modules():
                if isinstance(module, QuantizeLinear):
                    quant_layers.append((name, module))

        if training_args.use_lm_eval:
            if simple_evaluate is None or HFLM is None:
                raise ImportError("lm_eval is not installed. Please install it to enable use_lm_eval.")
            if training_args.lm_eval_tasks is None:
                raise ValueError("lm_eval_tasks must be provided when use_lm_eval is True.")

            if not is_main_process:
                log.info("Skipping lm_eval on non-zero ranks; rank 0 will run evaluation.")
            else:
                if has_eval_bit_list and len(quant_layers) == 0:
                    log.warning("eval_bit_list provided but no QuantizeLinear layers found; running a single lm_eval pass.")

                original_states = {}
                for name, layer in quant_layers:
                    original_states[name] = {
                        "multiple_bits_random_assign": getattr(layer, "multiple_bits_random_assign", None),
                        "cur_w_bits": getattr(layer, "cur_w_bits", None),
                    }

                model.eval()
                all_metrics = {}
                bit_targets = model_args.eval_bit_list if has_eval_bit_list and len(quant_layers) > 0 else [None]

                def run_lm_eval_once():
                    lm_batch_size = training_args.lm_eval_batch_size or training_args.per_device_eval_batch_size
                    lm_model = HFLM(
                        pretrained=model,
                        tokenizer=tokenizer,
                        batch_size=lm_batch_size,
                        trust_remote_code=True,
                    )
                    eval_kwargs = {
                        "model": lm_model,
                        "tasks": training_args.lm_eval_tasks,
                        "num_fewshot": training_args.lm_eval_num_fewshot,
                    }
                    if training_args.lm_eval_limit is not None:
                        eval_kwargs["limit"] = training_args.lm_eval_limit
                    return simple_evaluate(**eval_kwargs)

                for w_bits in bit_targets:
                    if w_bits is not None:
                        log.info(f"Evaluating {w_bits}-bit with lm_eval...")
                        for name, layer in quant_layers:
                            try:
                                if w_bits not in layer.w_bits_list:
                                    log.warning(
                                        f"Bit width {w_bits} not in layer {name}'s w_bits_list {layer.w_bits_list}, skipping this layer"
                                    )
                                    continue
                                layer.set_bits(w_bits)
                                if hasattr(layer, "multiple_bits_random_assign"):
                                    layer.multiple_bits_random_assign = False
                            except ValueError as e:
                                log.warning(f"Failed to set {w_bits}-bit for layer {name}: {e}")
                                continue

                    lm_results = run_lm_eval_once()
                    log.info(f"lm_eval results: {lm_results}")
                    bit_metrics = _flatten_lm_eval_results(lm_results, w_bits)
                    all_metrics.update(bit_metrics)
                    log.info(f"lm_eval results for {w_bits if w_bits is not None else 'default'}-bit: {bit_metrics}")

                for name, layer in quant_layers:
                    if name in original_states:
                        orig_state = original_states[name]
                        if orig_state.get("multiple_bits_random_assign") is not None:
                            layer.multiple_bits_random_assign = orig_state["multiple_bits_random_assign"]
                        if orig_state.get("cur_w_bits") is not None:
                            layer.cur_w_bits = orig_state["cur_w_bits"]

                trainer.log_metrics("lm_eval", all_metrics)
                trainer.save_metrics("lm_eval", all_metrics)
        elif not has_eval_dataset:
            log.warning("do_eval=True but no evaluation dataset was provided; skipping Trainer-based evaluation.")
        else:
            # Check if multi-bit evaluation is requested
            if has_eval_bit_list:
                # Multi-bit evaluation: evaluate on each bit width
                if is_main_process:
                    log.info(f"Starting multi-bit evaluation on bit widths: {model_args.eval_bit_list}")

                if len(quant_layers) == 0:
                    if is_main_process:
                        log.warning("No QuantizeLinear layers found in model, falling back to standard evaluation")
                    # Fall back to standard evaluation
                    metrics = trainer.evaluate()
                    sample_count = _sample_count_from_metrics(metrics)
                    if sample_count is not None:
                        metrics["eval_samples"] = sample_count
                    try:
                        perplexity = math.exp(metrics["eval_loss"])
                    except OverflowError:
                        perplexity = float("inf")
                    metrics["perplexity"] = perplexity
                    trainer.log_metrics("eval", metrics)
                    trainer.save_metrics("eval", metrics)
                else:
                    # Save original states for all layers
                    original_states = {}
                    for name, layer in quant_layers:
                        original_states[name] = {
                            'multiple_bits_random_assign': layer.multiple_bits_random_assign,
                            'cur_w_bits': layer.cur_w_bits,
                        }

                    # Set model to eval mode
                    model.eval()

                    all_metrics = {}

                    # Evaluate on each bit width
                    for w_bits in model_args.eval_bit_list:
                        if is_main_process:
                            log.info(f"Evaluating at {w_bits}-bit...")

                        # Set bit width and disable random assignment for all layers
                        for name, layer in quant_layers:
                            try:
                                # Check if w_bits is in the layer's w_bits_list
                                if w_bits not in layer.w_bits_list:
                                    if is_main_process:
                                        log.warning(f"Bit width {w_bits} not in layer {name}'s w_bits_list {layer.w_bits_list}, skipping this layer")
                                    continue
                                layer.set_bits(w_bits)
                                layer.multiple_bits_random_assign = False
                            except ValueError as e:
                                if is_main_process:
                                    log.warning(f"Failed to set {w_bits}-bit for layer {name}: {e}")
                                continue

                        # Run evaluation
                        try:
                            with torch.no_grad():
                                metrics = trainer.evaluate(eval_dataset=valid_data)

                            # Calculate perplexity
                            eval_loss = metrics.get("eval_loss", float("inf"))
                            try:
                                perplexity = math.exp(eval_loss)
                            except OverflowError:
                                perplexity = float("inf")

                            # Store metrics with bit-specific names
                            all_metrics[f"eval_loss_{w_bits}bit"] = eval_loss
                            all_metrics[f"perplexity_{w_bits}bit"] = perplexity
                            sample_count = _sample_count_from_metrics(metrics)
                            if sample_count is not None:
                                all_metrics[f"eval_samples_{w_bits}bit"] = sample_count

                            if is_main_process:
                                log.info(f"{w_bits}-bit evaluation: eval_loss={eval_loss:.4f}, perplexity={perplexity:.4f}")

                        except Exception as e:
                            if is_main_process:
                                log.error(f"Error during evaluation at {w_bits}-bit: {e}")
                            continue

                    # Restore original states
                    for name, layer in quant_layers:
                        if name in original_states:
                            orig_state = original_states[name]
                            layer.multiple_bits_random_assign = orig_state['multiple_bits_random_assign']
                            layer.cur_w_bits = orig_state['cur_w_bits']

                    # Log and save all metrics
                    if is_main_process:
                        log.info(f"Completed multi-bit evaluation. Results: {all_metrics}")

                    trainer.log_metrics("eval", all_metrics)
                    trainer.save_metrics("eval", all_metrics)
            else:
                # Standard single-bit evaluation
                metrics = trainer.evaluate()
                sample_count = _sample_count_from_metrics(metrics)
                if sample_count is not None:
                    metrics["eval_samples"] = sample_count
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
