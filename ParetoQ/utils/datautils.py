# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


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

import json
import logging
from typing import Optional
import torch

log = logging.getLogger(__name__)

class StreamingJsonlDataset(torch.utils.data.IterableDataset):
    """
    Stream jsonl, tokenize on-the-fly, and pack into block_size chunks.
    Optionally shards lines across ranks by modulo to avoid duplication.
    添加了token遍历统计功能，可以实时打印已处理的token数量
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        print_interval: int = 100000,  # 每处理10万个token打印一次
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.max_samples = max_samples if max_samples and max_samples > 0 else None
        self.print_interval = print_interval
        
        # Token统计相关变量
        self.total_processed_tokens = 0  # 已处理的总token数
        self.total_yielded_tokens = 0    # 已产出的总token数（按block_size分块的）
        self._line_token_count = 0       # 每行处理的token数（临时）

    def __iter__(self):
        buffer_ids = []
        yielded = 0
        # 重置统计变量（支持多次迭代）
        self.total_processed_tokens = 0
        self.total_yielded_tokens = 0
        
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if (idx % self.world_size) != self.rank:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse line %s in %s: %s", idx, self.path, exc)
                    continue

                if not text:
                    continue

                # 分词并统计token数
                tokens = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
                
                # 更新已处理的token总数
                self._line_token_count = len(tokens)
                self.total_processed_tokens += self._line_token_count
                buffer_ids.extend(tokens)

                # 按block_size分块并产出
                while len(buffer_ids) >= self.block_size:
                    chunk = buffer_ids[: self.block_size]
                    buffer_ids = buffer_ids[self.block_size :]
                    yielded += 1
                    self.total_yielded_tokens += len(chunk)  # 统计已产出的token数
                    
                    # 按间隔打印统计信息
                    if self.total_processed_tokens % self.print_interval < self._line_token_count:
                        self._print_token_stats(idx)
                    
                    yield dict(
                        input_ids=torch.tensor(chunk, dtype=torch.long),
                        labels=torch.tensor(chunk, dtype=torch.long),
                    )
                    
                    if self.max_samples and yielded >= self.max_samples:
                        # 最后打印一次统计信息
                        self._print_token_stats(idx, final=True)
                        return
        
        # 遍历结束后打印最终统计
        self._print_token_stats(idx, final=True)

    def _print_token_stats(self, line_idx: int, final: bool = False):
        """
        打印token统计信息
        :param line_idx: 当前处理的行号
        :param final: 是否是最终统计
        """
        status = "Final" if final else "Current"
        log.info(
            f"{status} Token Statistics | "
            f"Rank: {self.rank} | "
            f"Processed Lines: {line_idx + 1} | "
            f"Total Processed Tokens: {self.total_processed_tokens:,} | "
            f"Total Yielded Tokens: {self.total_yielded_tokens:,} | "
            f"Remaining in Buffer: {len(self.__dict__.get('buffer_ids', [])):,}"
        )
    
    def get_token_counts(self):
        """
        获取当前的token统计信息（供外部调用）
        :return: 包含统计信息的字典
        """
        return {
            "total_processed_tokens": self.total_processed_tokens,
            "total_yielded_tokens": self.total_yielded_tokens,
            "remaining_in_buffer": len(self.__dict__.get('buffer_ids', [])),
            "rank": self.rank
        }


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
