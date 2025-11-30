# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Sequence, Iterator, Optional

import numpy as np
import torch
from datasets import load_dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None, train_dataset_name=None, train_dataset_subset=None, use_streaming=True):
    """
    Load training and validation datasets.
    
    Supports two modes:
    1. Local file mode: If train_path is a local file path, load from JSONL file
    2. Streaming mode: If train_dataset_name is specified, load from HuggingFace dataset with streaming
    
    Args:
        train_path: Local file path (for backward compatibility) or None
        valid_path: Local validation file path or None
        train_dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        train_dataset_subset: Dataset subset name (optional)
        use_streaming: Whether to use streaming mode (default: True)
    
    Returns:
        train_data, valid_data: Dataset objects (list for local mode, iterator for streaming mode)
    """
    # Determine which mode to use
    use_local_file = False
    if train_path is not None:
        # Check if train_path is a valid local file
        if os.path.isfile(train_path) and train_path.endswith('.jsonl'):
            use_local_file = True
    
    # Use streaming mode if dataset name is specified and not using local file
    if train_dataset_name is not None and not use_local_file and use_streaming:
        # Streaming mode: return None for train_data (will be handled by StreamingJsonDataset)
        # For validation, still try to load from local file if provided
        valid_data = []
        if valid_path and os.path.isfile(valid_path) and valid_path.endswith('.jsonl'):
            f = open(valid_path, "r", encoding="utf-8")
            while True:
                line = f.readline()
                if not line:
                    break
                valid_data.append(json.loads(line))
            f.close()
        return None, valid_data if valid_data else None
    else:
        # Local file mode (backward compatible)
        if train_path is None:
            raise ValueError("Either train_path or train_dataset_name must be provided")
        
        f = open(train_path, "r", encoding="utf-8")
        data = []
        while True:
            line = f.readline()
            if not line:
                break
            data.append(json.loads(line))
        f.close()
        train_data = []
        valid_data = []
        if valid_path:
            f = open(valid_path, "r", encoding="utf-8")
            while True:
                line = f.readline()
                if not line:
                    break
                valid_data.append(json.loads(line))
            f.close()
            train_data = data
        else:
            train_data = data[10000:]
            valid_data = data[:10000]
        return train_data, valid_data


class StreamingJsonDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset that loads data from HuggingFace datasets on-the-fly.
    Supports infinite looping and distributed training.
    """
    def __init__(self, dataset_name, tokenizer, block_size=1024, subset=None, infinite=True):
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.infinite = infinite
        
    def _create_dataset(self):
        """Create a HuggingFace dataset with proper sharding for distributed training."""
        dataset_kwargs = {
            "streaming": True,
            "split": "train",
        }
        if self.subset is not None:
            dataset_kwargs["name"] = self.subset
        
        dataset = load_dataset(self.dataset_name, **dataset_kwargs)
        
        # Handle distributed training sharding
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            # Shard the dataset for this process
            dataset = dataset.shard(num_shards=world_size, index=rank)
        
        return dataset
    
    def _get_data_iterator(self):
        """Get data iterator with infinite looping support."""
        while True:
            dataset = self._create_dataset()
            for sample in dataset:
                text = sample.get('text', '').strip()
                if text:
                    yield {"text": text}
            
            if not self.infinite:
                break
    
    def _tokenize_and_group(self, text_dict, token_buffer, label_buffer):
        """Tokenize a single text and add to buffer, yielding chunks when ready."""
        tokenized = self.tokenizer(text_dict["text"])
        input_ids = tokenized["input_ids"]
        
        # Add to buffer
        token_buffer.extend(input_ids)
        label_buffer.extend(input_ids)
        
        # Yield chunks when buffer is large enough
        while len(token_buffer) >= self.block_size:
            chunk_input_ids = token_buffer[:self.block_size]
            chunk_labels = label_buffer[:self.block_size]
            token_buffer[:] = token_buffer[self.block_size:]
            label_buffer[:] = label_buffer[self.block_size:]
            
            yield {
                "input_ids": chunk_input_ids,
                "labels": chunk_labels
            }
    
    def __iter__(self):
        """Iterate over the dataset, yielding tokenized and grouped samples."""
        token_buffer = []
        label_buffer = []
        
        data_iter = self._get_data_iterator()
        
        for text_dict in data_iter:
            for chunk in self._tokenize_and_group(text_dict, token_buffer, label_buffer):
                yield chunk
    
    def __len__(self):
        """Return None for streaming datasets (size is unknown)."""
        return None


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict
