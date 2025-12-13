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
    def __init__(
        self,
        dataset,
        tokenizer,
        block_size=1024,
        cache_dir: Optional[str] = None,
        data_file_path: Optional[str] = None,
        streaming: bool = False,
        batch_size_samples: int = 1000,
        shard_size_tokens: int = 10_000_000,
        max_samples: Optional[int] = None,
        force_retokenize: bool = False,
        flush_each_batch: bool = True,
    ):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = cache_dir or "./cache"
        self.data_file_path = data_file_path
        self.streaming = streaming and data_file_path is not None
        self.batch_size_samples = batch_size_samples
        self.shard_size_tokens = shard_size_tokens
        self.max_samples = max_samples if max_samples is not None and max_samples > 0 else None
        self.force_retokenize = force_retokenize
        self.flush_each_batch = flush_each_batch

        self.sharded_manifest = None
        self.input_ids = None
        self.labels = None
        self.data = None
        self.cache_key = None
        self._streaming_num_chunks = 0
        self._manifest_path = None

        # Try to load from pretokenize.py cache if data_file_path is provided
        cache_path = None
        if self.data_file_path is not None:
            self.cache_key = self._generate_cache_key_from_file(data_file_path, tokenizer, block_size)
            manifest_path = os.path.join(self.cache_dir, f"pretokenized_{self.cache_key}_manifest.json")
            cache_path = os.path.join(self.cache_dir, f"pretokenized_{self.cache_key}.pkl")
            self._manifest_path = manifest_path

            if self.force_retokenize:
                for old_path in (manifest_path, cache_path):
                    if old_path and os.path.exists(old_path):
                        logging.info(f"Force re-tokenize: removing existing cache {old_path}")
                        os.remove(old_path)

            if os.path.exists(manifest_path):
                logging.info(f"Streaming dataset from sharded pretokenize cache: {manifest_path}")
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    self.sharded_manifest = json.load(mf)
                self.block_size = self.sharded_manifest.get("block_size", block_size)
                self._streaming_num_chunks = self.sharded_manifest.get("num_chunks", 0)
                return

            if os.path.exists(cache_path) and not self.streaming:
                # Legacy single-file cache path
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

        # If streaming is enabled, defer tokenization to __iter__ to overlap with training
        if self.streaming:
            logging.info(
                "Using streaming tokenization. Batches will be tokenized, cached to disk, and yielded immediately."
            )
            return

        # Process data eagerly if cache doesn't exist and streaming is disabled
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
        if self.sharded_manifest:
            num_chunks = self.sharded_manifest.get("num_chunks", None)
            if num_chunks is None:
                num_chunks = sum(s.get("num_chunks", 0) for s in self.sharded_manifest.get("shards", []))
            return num_chunks
        if self._streaming_num_chunks:
            return self._streaming_num_chunks
        if self.input_ids is None:
            return 0
        return len(self.input_ids)

    def __getitem__(self, i):
        if self.sharded_manifest or self.streaming:
            raise IndexError("Random access is not supported for sharded streaming datasets.")
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        if self.sharded_manifest:
            return self._iter_sharded()
        if self.streaming and self.data_file_path is not None:
            return self._iter_streaming_and_cache()
        if self.data is None:
            return iter([])
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

    def _iter_sharded(self):
        """Stream shards one by one to avoid loading the full token buffer into memory."""
        shards = self.sharded_manifest.get("shards", [])
        if not shards:
            logging.warning("Sharded manifest is empty, nothing to iterate.")
            return
        for shard in shards:
            shard_path = shard.get("path")
            if shard_path is None:
                logging.warning("Shard entry missing path, skipping.")
                continue
            full_path = shard_path if os.path.isabs(shard_path) else os.path.join(self.cache_dir, shard_path)
            if not os.path.exists(full_path):
                logging.warning(f"Shard file not found: {full_path}, skipping.")
                continue
            with open(full_path, "rb") as f:
                cached_data = pickle.load(f)
            input_ids = cached_data.get("input_ids", [])
            labels = cached_data.get("labels", [])
            for i in range(len(input_ids)):
                yield dict(input_ids=input_ids[i], labels=labels[i])
            del input_ids, labels, cached_data

    def _iter_streaming_and_cache(self):
        """Tokenize the source file in batches, cache shards to disk, and yield chunks immediately."""
        if self.data_file_path is None:
            logging.warning("Streaming enabled but no data_file_path provided; yielding nothing.")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = self.cache_key or self._generate_cache_key_from_file(
            self.data_file_path, self.tokenizer, self.block_size
        )
        manifest_path = self._manifest_path or os.path.join(
            self.cache_dir, f"pretokenized_{cache_key}_manifest.json"
        )

        all_input_ids = []
        all_labels = []
        shard_id = 0
        shard_samples = 0
        shard_tokens = 0
        shards = []
        batch_data = []
        total_tokens = 0
        sample_count = 0

        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if self.max_samples and sample_count >= self.max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    if "text" not in data:
                        logging.warning(f"Line {line_num}: Missing 'text' field, skipping")
                        continue
                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
                    continue

                batch_data.append(data)
                sample_count += 1

                if len(batch_data) >= self.batch_size_samples:
                    batch_tokens, batch_chunks = self._process_batch(
                        batch_data, all_input_ids, all_labels
                    )
                    total_tokens += batch_tokens
                    shard_tokens += batch_tokens
                    shard_samples += len(batch_data)
                    batch_data = []

                    current_buffer_tokens = len(all_input_ids) * self.block_size
                    if current_buffer_tokens >= self.shard_size_tokens or (
                        self.flush_each_batch and batch_chunks > 0
                    ):
                        shard_meta = self._flush_shard(
                            cache_key=cache_key,
                            shard_id=shard_id,
                            input_ids=all_input_ids,
                            labels=all_labels,
                            shard_tokens=shard_tokens,
                            shard_samples=shard_samples,
                        )
                        shards.append(shard_meta)
                        self._streaming_num_chunks += shard_meta["num_chunks"]
                        shard_id += 1
                        shard_tokens = 0
                        shard_samples = 0
                        all_input_ids = []
                        all_labels = []
                        yield from self._yield_shard(shard_meta)

        # Process remaining batch
        if batch_data:
            batch_tokens, batch_chunks = self._process_batch(batch_data, all_input_ids, all_labels)
            total_tokens += batch_tokens
            shard_tokens += batch_tokens
            shard_samples += len(batch_data)

        # Final flush
        if all_input_ids:
            shard_meta = self._flush_shard(
                cache_key=cache_key,
                shard_id=shard_id,
                input_ids=all_input_ids,
                labels=all_labels,
                shard_tokens=shard_tokens,
                shard_samples=shard_samples,
            )
            shards.append(shard_meta)
            self._streaming_num_chunks += shard_meta["num_chunks"]
            yield from self._yield_shard(shard_meta)

        manifest = {
            "version": 1,
            "cache_key": cache_key,
            "block_size": self.block_size,
            "num_samples": sample_count,
            "num_tokens": total_tokens,
            "num_chunks": self._streaming_num_chunks,
            "shard_size_tokens": self.shard_size_tokens,
            "shards": shards,
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        self.sharded_manifest = manifest

    def _process_batch(self, batch_data, all_input_ids, all_labels):
        """Process a batch of samples and append to all_input_ids/all_labels."""
        tokenized_batch = []
        for data in batch_data:
            try:
                tokenized = self.tokenizer(data["text"])
                tokenized_batch.append(tokenized)
            except Exception as e:
                logging.warning(f"Tokenization error: {e}, skipping sample")
                continue

        if not tokenized_batch:
            return 0, 0

        concatenated = {"input_ids": []}
        for tokenized in tokenized_batch:
            concatenated["input_ids"].extend(tokenized["input_ids"])

        total_length = len(concatenated["input_ids"])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size

        num_chunks = 0
        for i in range(0, total_length, self.block_size):
            chunk = concatenated["input_ids"][i : i + self.block_size]
            all_input_ids.append(chunk)
            all_labels.append(chunk.copy())
            num_chunks += 1

        return total_length, num_chunks

    def _flush_shard(
        self,
        cache_key: str,
        shard_id: int,
        input_ids,
        labels,
        shard_tokens: int,
        shard_samples: int,
    ):
        """Persist current buffers to a shard file."""
        shard_filename = f"pretokenized_{cache_key}_part{shard_id:05d}.pkl"
        shard_path = os.path.join(self.cache_dir, shard_filename)
        shard_data = {
            "input_ids": input_ids,
            "labels": labels,
            "num_samples": shard_samples,
            "num_tokens": shard_tokens,
            "num_chunks": len(input_ids),
            "block_size": self.block_size,
        }
        with open(shard_path, "wb") as f:
            pickle.dump(shard_data, f)
        logging.info(
            f"[streaming] Flushed shard {shard_id} with {len(input_ids)} chunks "
            f"({shard_tokens:,} tokens, {shard_samples} samples) -> {shard_path}"
        )
        return {
            "path": shard_filename,
            "num_samples": shard_samples,
            "num_tokens": shard_tokens,
            "num_chunks": len(input_ids),
        }

    def _yield_shard(self, shard_meta):
        """Yield chunks from a shard file and free memory after use."""
        shard_path = shard_meta.get("path")
        if shard_path is None:
            return
        full_path = shard_path if os.path.isabs(shard_path) else os.path.join(self.cache_dir, shard_path)
        if not os.path.exists(full_path):
            logging.warning(f"Shard file not found when yielding: {full_path}")
            return
        with open(full_path, "rb") as f:
            cached_data = pickle.load(f)
        input_ids = cached_data.get("input_ids", [])
        labels = cached_data.get("labels", [])
        for i in range(len(input_ids)):
            yield dict(input_ids=input_ids[i], labels=labels[i])
        del input_ids, labels, cached_data


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict
