# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None):
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


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size=1024, cache_dir: Optional[str] = None, data_file_path: Optional[str] = None):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.data_file_path = data_file_path
        
        # Try to load from pretokenize.py cache if data_file_path is provided
        cache_path = None
        if cache_dir is not None and data_file_path is not None:
            cache_key = self._generate_cache_key_from_file(data_file_path, tokenizer, block_size)
            cache_path = os.path.join(cache_dir, f"pretokenized_{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                logging.info(f"Loading dataset from pretokenize cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.input_ids = cached_data["input_ids"]
                    self.labels = cached_data["labels"]
                    self.data = [
                        dict(input_ids=self.input_ids[i], labels=self.labels[i])
                        for i in range(len(self.input_ids))
                    ]
                    logging.info(f"Successfully loaded {len(self.input_ids)} samples from pretokenize cache")
                    return
        
        # Process data if cache doesn't exist
        logging.info("Tokenizing and grouping dataset...")
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
    
    def _generate_cache_key_from_file(self, file_path, tokenizer, block_size):
        """Generate cache key based on file path, tokenizer, and block_size (compatible with pretokenize.py)."""
        # Get file stats for cache key
        file_stat = os.stat(file_path)
        file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        
        # Get tokenizer info
        tokenizer_str = ""
        if hasattr(tokenizer, "name_or_path"):
            tokenizer_str = tokenizer.name_or_path
        elif hasattr(tokenizer, "__class__"):
            tokenizer_str = tokenizer.__class__.__name__
        
        cache_string = f"{file_info}_{tokenizer_str}_{block_size}"
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

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