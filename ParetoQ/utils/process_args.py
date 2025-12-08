# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, List

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    w_bits: Optional[int] = field(
        default=32,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    w_bits_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of bit widths for multi-bit training, e.g., '1,2,4'. If provided, enables multi-bit training."
        },
    )
    prob_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of probabilities for weighted bit selection, e.g., '3,1,1'. Must match length of w_bits_list. If None or all equal, uses uniform distribution."
        },
    )
    contain_weight_clip_val: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set contain_weight_clip_val=True when load a trained quantized model."
        },
    )
    # Noise injection parameters
    noise_injection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to enable noise injection during quantization."},
    )
    noise_sigma_weights: Optional[float] = field(
        default=0.001,
        metadata={"help": "Standard deviation of noise for weights."},
    )
    noise_sigma_clipvals: Optional[float] = field(
        default=0.001,
        metadata={"help": "Standard deviation of noise for clip values."},
    )
    initialize_noise: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to initialize noise parameters."},
    )
    pre_quantization_noise: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to inject noise before quantization."},
    )
    post_quantization_noise: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to inject noise after quantization."},
    )
    trainable_noise_scale: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use trainable noise scale."},
    )
    # Multi-bit training parameters
    multiple_bits_random_assign: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to randomly assign bit widths during training."},
    )
    multiple_bits_random_assign_prob: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability of random bit assignment."},
    )
    multiple_bits_share_clipvals: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to share clip values across different bit widths."},
    )
    multiple_bits_disable_clipvals: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable clip values for multi-bit training."},
    )
    # Stretch quantization parameters
    use_stretch: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use stretch quantization."},
    )
    stretch_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "Alpha parameter for stretch quantization."},
    )
    # MobileLLM specific parameters
    share_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to share input and output embeddings for MobileLLM models."},
    )
    layer_sharing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to share layers (repeat each decoder layer twice) for MobileLLM models."},
    )
    eval_bit_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of bit widths for evaluation, e.g., '1,2,4'. If provided, evaluates model at each bit width during do_eval."
        },
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )
    streaming: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, tokenize on-the-fly with a streaming IterableDataset to avoid upfront tokenization."
        },
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)
    use_lm_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, run evaluation with lm_eval's HFLM + simple_evaluate instead of Trainer.evaluate."
        },
    )
    lm_eval_tasks: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated lm_eval task names to run (e.g., 'wikitext,hellaswag'). Required when use_lm_eval is True."
        },
    )
    lm_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size used by lm_eval. Defaults to per_device_eval_batch_size if not set."},
    )
    lm_eval_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Optional limit on the number of evaluation examples per task for lm_eval."},
    )
    lm_eval_num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "Number of few-shot examples used by lm_eval tasks."},
    )


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(model_args.local_dir, exist_ok=True)

    assert model_args.output_model_local_path is None

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, "models", str(model_args.output_model_filename)
    )
    
    # Parse w_bits_list from comma-separated string to list of integers
    if model_args.w_bits_list is not None:
        model_args.w_bits_list = [int(x.strip()) for x in model_args.w_bits_list.split(',')]
    else:
        model_args.w_bits_list = None
    
    # Parse prob_list from comma-separated string to list of floats
    if model_args.prob_list is not None:
        model_args.prob_list = [float(x.strip()) for x in model_args.prob_list.split(',')]
        # Validate that prob_list is only used when w_bits_list is provided
        if model_args.w_bits_list is None:
            raise ValueError("prob_list can only be used when w_bits_list is provided")
        # Validate that prob_list matches w_bits_list length
        if len(model_args.prob_list) != len(model_args.w_bits_list):
            raise ValueError(f"prob_list length ({len(model_args.prob_list)}) must match w_bits_list length ({len(model_args.w_bits_list)})")
    else:
        model_args.prob_list = None
    
    # Parse eval_bit_list from comma-separated string to list of integers
    if model_args.eval_bit_list is not None:
        model_args.eval_bit_list = [int(x.strip()) for x in model_args.eval_bit_list.split(',')]
    else:
        model_args.eval_bit_list = None

    # Parse lm_eval_tasks from comma-separated string to list of task names
    if getattr(training_args, "lm_eval_tasks", None):
        training_args.lm_eval_tasks = [
            task.strip() for task in training_args.lm_eval_tasks.split(",") if task.strip()
        ]
        if len(training_args.lm_eval_tasks) == 0:
            training_args.lm_eval_tasks = None
    else:
        training_args.lm_eval_tasks = None

    # Normalize lm_eval_limit
    if getattr(training_args, "lm_eval_limit", None) is not None:
        if training_args.lm_eval_limit <= 0:
            training_args.lm_eval_limit = None

    return model_args, data_args, training_args
