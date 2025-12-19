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
    # Multi-bit training parameters
    multiple_bits_random_assign: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to randomly assign bit widths during training."},
    )
    multiple_bits_random_assign_prob: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability of random bit assignment."},
    )
    noise_injection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to inject noise during training."},
    )
    eval_bit_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of bit widths for evaluation, e.g., '1,2,4'. If provided, evaluates model at each bit width during do_eval."
        },
    )
    debug_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of layer names to enable debug mode, e.g., 'model.layers.0.self_attn.q_proj,model.layers.0.self_attn.k_proj'. If provided, enables debug printing for specified layers during training."
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
    dataset_name: str = field(
        default="HuggingFaceFW/fineweb-edu",
        metadata={"help": "Dataset identifier passed to datasets.load_dataset when streaming (e.g., HuggingFaceFW/fineweb-edu)."},
    )
    dataset_config_name: Optional[str] = field(
        default="sample-100BT",
        metadata={"help": "Optional dataset config/subset name (e.g., fineweb-edu crawl name)."},
    )
    dataset_split: str = field(
        default="train", metadata={"help": "Dataset split to use for training when streaming."}
    )
    eval_dataset_split: Optional[str] = field(
        default=None,
        metadata={"help": "Optional split to use for evaluation; falls back to dataset_split if not provided."},
    )
    streaming: bool = field(
        default=True, metadata={"help": "Use datasets streaming mode to avoid pre-downloading jsonl files."}
    )
    shuffle_buffer_size: int = field(
        default=10_000, metadata={"help": "Shuffle buffer size for streaming datasets."}
    )
    shuffle_seed: int = field(default=42, metadata={"help": "Shuffle seed for streaming datasets."})
    text_column: str = field(default="text", metadata={"help": "Text column to tokenize from the dataset."})
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
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
    
    # Parse debug_layers from comma-separated string to list of strings
    if model_args.debug_layers is not None:
        model_args.debug_layers = [x.strip() for x in model_args.debug_layers.split(',')]
    else:
        model_args.debug_layers = None

    return model_args, data_args, training_args
