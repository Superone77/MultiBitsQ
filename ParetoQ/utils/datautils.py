# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import IterableDataset, get_worker_info


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

log = logging.getLogger(__name__)

DEFAULT_VALID_SPLIT = 10_000


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def _infer_num_proc(num_proc: Optional[int] = None) -> int:
    """Decide how many worker processes to use for tokenization."""
    env_override = os.getenv("TOKENIZER_NUM_PROC")
    if env_override:
        try:
            env_value = int(env_override)
            if env_value > 0:
                return env_value
        except ValueError:
            log.warning("Invalid TOKENIZER_NUM_PROC value '%s', falling back to auto.", env_override)

    if num_proc is not None and num_proc > 0:
        return num_proc

    cpu_count = os.cpu_count() or 1
    # Cap at 16 to avoid oversubscription on very large machines
    return max(1, min(cpu_count, 16))


def _get_cache_dir(train_path: str, block_size: int, cache_dir: Optional[str]) -> Path:
    """Get a stable cache directory for tokenized datasets."""
    base_dir = Path(cache_dir) if cache_dir is not None else Path(train_path).parent
    cache_dir = base_dir / f"{Path(train_path).stem}_tok_bs{block_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_raw_datasets(
    train_path: str, valid_path: Optional[str] = None, valid_size: int = DEFAULT_VALID_SPLIT
) -> Tuple[object, Optional[object]]:
    """Load raw jsonl data with datasets; fall back to a small held-out split when no eval set is provided."""
    data_files = {"train": train_path}
    if valid_path is not None:
        data_files["validation"] = valid_path

    raw = load_dataset("json", data_files=data_files)
    train_ds = raw["train"]

    if valid_path is not None:
        return train_ds, raw["validation"]

    if valid_size <= 0 or valid_size >= train_ds.num_rows:
        log.warning(
            "Requested validation split of %s rows is not possible (train rows=%s); using empty validation set.",
            valid_size,
            train_ds.num_rows,
        )
        return train_ds, None

    split = train_ds.train_test_split(test_size=valid_size)
    return split["train"], split["test"]


def _tokenize_and_group(
    dataset,
    tokenizer,
    block_size: int,
    num_proc: Optional[int],
    desc_prefix: str,
):
    """Tokenize text and pack into fixed-length blocks."""
    num_proc = _infer_num_proc(num_proc)
    column_names = dataset.column_names

    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            add_special_tokens=True,
            return_attention_mask=False,
            padding=False,
            truncation=False,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"{desc_prefix}: tokenizing with {num_proc} workers",
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        result = {
            k: [
                t[i : i + block_size] for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
        load_from_cache_file=True,
        desc=f"{desc_prefix}: grouping into blocks of {block_size}",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "labels"])
    return tokenized


def prepare_tokenized_datasets(
    train_path: str,
    valid_path: Optional[str],
    tokenizer,
    block_size: int,
    cache_dir: Optional[str],
    num_proc: Optional[int],
    max_train_samples: int,
    max_eval_samples: int,
    rank: int,
    world_size: int,
    valid_size: int = DEFAULT_VALID_SPLIT,
):
    """
    Tokenize the dataset once on rank 0, cache it to disk, and let all ranks load from the cache.
    This avoids repeating a costly tokenization pass per process for large corpora.
    """
    cache_root = _get_cache_dir(train_path, block_size, cache_dir)
    train_cache = cache_root / "train"
    eval_cache = cache_root / "validation"

    # Only build the cache on a single worker
    if rank == 0 and not train_cache.exists():
        log.info("No cached tokenized dataset found. Building cache at %s", cache_root)
        raw_train, raw_valid = _load_raw_datasets(train_path, valid_path, valid_size)
        tokenized_train = _tokenize_and_group(
            raw_train, tokenizer, block_size, num_proc=num_proc, desc_prefix="train"
        )
        tokenized_train.save_to_disk(train_cache)

        if raw_valid is not None:
            tokenized_valid = _tokenize_and_group(
                raw_valid,
                tokenizer,
                block_size=min(block_size, 1024),
                num_proc=num_proc,
                desc_prefix="validation",
            )
            tokenized_valid.save_to_disk(eval_cache)
    elif rank == 0:
        log.info("Using existing tokenized cache at %s", cache_root)

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    train_dataset = load_from_disk(train_cache)
    if max_train_samples and max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))

    eval_dataset = None
    if valid_path is not None or eval_cache.exists():
        if eval_cache.exists():
            eval_dataset = load_from_disk(eval_cache)
            if max_eval_samples and max_eval_samples > 0:
                eval_dataset = eval_dataset.select(
                    range(min(max_eval_samples, len(eval_dataset)))
                )
        else:
            log.warning(
                "Requested evaluation set but no cached validation data found at %s. Continuing without eval data.",
                eval_cache,
            )

    return train_dataset, eval_dataset

def _normalize_max_samples(value: Optional[int]) -> Optional[int]:
    if value is None or value < 0:
        return None
    return value


class StreamingTokenizedDataset(IterableDataset):
    """
    Streaming dataset wrapper that tokenizes examples and packs them into fixed block_size chunks.
    Works with Hugging Face's streaming datasets and supports worker-level sharding.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        block_size: int,
        text_column: str = "text",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.max_samples = _normalize_max_samples(max_samples)

    def __len__(self):
        if self.max_samples is None:
            raise TypeError("Streaming dataset does not expose a stable length.")
        return self.max_samples

    def __iter__(self):
        worker = get_worker_info()
        iterator = iter(self.dataset)
        if worker is not None:
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        buffer_ids = []
        yielded = 0
        for example in iterator:
            if isinstance(example, dict):
                text = example.get(self.text_column, "")
            else:
                text = getattr(example, self.text_column, "")

            if not text:
                continue

            token_ids = self.tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            buffer_ids.extend(token_ids)

            while len(buffer_ids) >= self.block_size:
                chunk = buffer_ids[: self.block_size]
                buffer_ids = buffer_ids[self.block_size :]
                yielded += 1

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }

                if self.max_samples is not None and yielded >= self.max_samples:
                    return


def build_streaming_text_dataset(
    *,
    tokenizer,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    block_size: int,
    rank: int,
    world_size: int,
    streaming: bool = True,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    text_column: str = "text",
    max_samples: Optional[int] = None,
    data_files: Optional[str] = None,
):
    load_kwargs = {"split": split, "streaming": streaming}
    if data_files is not None:
        load_kwargs["data_files"] = data_files

    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    if shuffle:
        try:
            dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
        except TypeError:
            dataset = dataset.shuffle(seed=shuffle_seed)
    if world_size > 1:
        dataset = datasets.distributed.split_dataset_by_node(
            dataset, rank=rank, world_size=world_size
        )

    return StreamingTokenizedDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        block_size=block_size,
        text_column=text_column,
        max_samples=max_samples,
    )


def get_dataset_length(dataset) -> Optional[int]:
    try:
        return len(dataset)
    except TypeError:
        return None


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict
