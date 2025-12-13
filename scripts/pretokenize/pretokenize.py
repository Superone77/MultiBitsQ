# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pretokenize script for large-scale datasets.
This script processes JSONL files in a streaming fashion to avoid loading
all data into memory at once, making it suitable for processing 100B+ token datasets.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from typing import Optional

import torch
import transformers

# Try to import utils, fallback to basic logging if not available
try:
    from utils import utils
    log = utils.get_logger("pretokenize")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("pretokenize")


def estimate_memory_requirements(num_tokens: int, block_size: int = 2048, dtype: str = "int32"):
    """
    Estimate memory requirements for tokenized dataset.
    
    Args:
        num_tokens: Total number of tokens
        block_size: Block size for chunking
        dtype: Data type for tokens (int32 or int16)
    
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_token = 4 if dtype == "int32" else 2
    
    # Memory for input_ids and labels (same size)
    tokens_memory_gb = (num_tokens * bytes_per_token * 2) / (1024 ** 3)
    
    # Memory for chunked data (overhead for list of lists)
    num_chunks = num_tokens // block_size
    chunk_overhead_gb = (num_chunks * 8 * 2) / (1024 ** 3)  # 8 bytes per pointer, 2 for ids and labels
    
    # Python object overhead (rough estimate: 10% overhead)
    python_overhead_gb = tokens_memory_gb * 0.1
    
    total_memory_gb = tokens_memory_gb + chunk_overhead_gb + python_overhead_gb
    
    return {
        "tokens_memory_gb": tokens_memory_gb,
        "chunk_overhead_gb": chunk_overhead_gb,
        "python_overhead_gb": python_overhead_gb,
        "total_memory_gb": total_memory_gb,
        "recommended_batch_size_tokens": min(10_000_000, num_tokens // 100)  # Process in batches
    }


def process_data_streaming(
    input_path: str,
    tokenizer,
    block_size: int,
    cache_dir: str,
    batch_size_samples: int = 1000,
    max_samples: Optional[int] = None,
    shard_size_tokens: int = 10_000_000,
):
    """
    Process dataset in streaming fashion to avoid memory issues.
    
    Args:
        input_path: Path to input JSONL file
        tokenizer: Tokenizer instance
        block_size: Block size for chunking
        cache_dir: Directory to save cache files
        batch_size_samples: Number of samples to process in each batch
        max_samples: Maximum number of samples to process (None for all)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on file and tokenizer
    cache_key = _generate_cache_key_from_file(input_path, tokenizer, block_size)
    manifest_path = os.path.join(cache_dir, f"pretokenized_{cache_key}_manifest.json")
    legacy_cache_path = os.path.join(cache_dir, f"pretokenized_{cache_key}.pkl")
    
    # Check if cache already exists (manifest or legacy single file)
    if os.path.exists(manifest_path):
        log.info(f"Sharded cache already exists: {manifest_path}")
        log.info("Use --force to regenerate cache")
        return manifest_path
    if os.path.exists(legacy_cache_path):
        log.info(f"Legacy cache already exists: {legacy_cache_path}")
        log.info("Use --force to regenerate cache")
        return legacy_cache_path
    
    log.info(f"Starting pretokenization of {input_path}")
    log.info(f"Block size: {block_size}, Batch size: {batch_size_samples} samples")
    
    all_input_ids = []
    all_labels = []
    shard_id = 0
    shard_samples = 0
    shard_tokens = 0
    shards = []
    batch_data = []
    total_tokens = 0
    sample_count = 0
    total_chunks = 0
    
    # Stream through the file
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and sample_count >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                if "text" not in data:
                    log.warning(f"Line {line_num}: Missing 'text' field, skipping")
                    continue
                
                batch_data.append(data)
                sample_count += 1
                
                # Process batch when it reaches batch_size_samples
                if len(batch_data) >= batch_size_samples:
                    batch_tokens = _process_batch(batch_data, tokenizer, block_size, 
                                                 all_input_ids, all_labels)
                    total_tokens += batch_tokens
                    shard_tokens += batch_tokens
                    shard_samples += len(batch_data)
                    batch_data = []
                    
                    if line_num % 10000 == 0:
                        log.info(f"Processed {line_num} lines, {sample_count} samples, "
                               f"~{total_tokens:,} tokens")
                    
                    # Flush shard if exceeding shard_size_tokens
                    current_buffer_tokens = len(all_input_ids) * block_size
                    if current_buffer_tokens >= shard_size_tokens:
                        shard_path = _flush_shard(
                            cache_dir, cache_key, shard_id, all_input_ids, all_labels,
                            block_size, shard_tokens, shard_samples
                        )
                        shards.append(shard_path)
                        total_chunks += shard_path["num_chunks"]
                        shard_id += 1
                        shard_tokens = 0
                        shard_samples = 0
                        all_input_ids = []
                        all_labels = []
            
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
                continue
            except Exception as e:
                log.warning(f"Line {line_num}: Unexpected error: {e}, skipping")
                continue
        
        # Process remaining batch
        if batch_data:
            batch_tokens = _process_batch(batch_data, tokenizer, block_size,
                                         all_input_ids, all_labels)
            total_tokens += batch_tokens
            shard_tokens += batch_tokens
            shard_samples += len(batch_data)
    
    # Final flush
    if all_input_ids:
        shard_path = _flush_shard(
            cache_dir, cache_key, shard_id, all_input_ids, all_labels,
            block_size, shard_tokens, shard_samples
        )
        shards.append(shard_path)
        total_chunks += shard_path["num_chunks"]
    
    log.info(f"Tokenization complete: {sample_count} samples, ~{total_tokens:,} tokens")
    log.info(f"Total chunks: {total_chunks}")
    
    # Save manifest
    manifest = {
        "version": 1,
        "cache_key": cache_key,
        "block_size": block_size,
        "num_samples": sample_count,
        "num_tokens": total_tokens,
        "num_chunks": total_chunks,
        "shard_size_tokens": shard_size_tokens,
        "shards": shards,
    }
    log.info(f"Saving manifest: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    
    log.info(f"Manifest saved successfully: {manifest_path}")
    return manifest_path


def _process_batch(batch_data, tokenizer, block_size, all_input_ids, all_labels):
    """Process a batch of samples and append to all_input_ids/all_labels."""
    tokenized_batch = []
    for data in batch_data:
        try:
            tokenized = tokenizer(data["text"])
            tokenized_batch.append(tokenized)
        except Exception as e:
            log.warning(f"Tokenization error: {e}, skipping sample")
            continue
    
    if not tokenized_batch:
        return 0
    
    # Group texts (concatenate and chunk)
    concatenated = {"input_ids": []}
    for tokenized in tokenized_batch:
        concatenated["input_ids"].extend(tokenized["input_ids"])
    
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split into chunks
    for i in range(0, total_length, block_size):
        chunk = concatenated["input_ids"][i:i + block_size]
        all_input_ids.append(chunk)
        all_labels.append(chunk.copy())
    
    return total_length


def _flush_shard(cache_dir, cache_key, shard_id, input_ids, labels, block_size, shard_tokens, shard_samples):
    """Persist current buffers to a shard file and return shard metadata."""
    shard_filename = f"pretokenized_{cache_key}_part{shard_id:05d}.pkl"
    shard_path = os.path.join(cache_dir, shard_filename)
    shard_data = {
        "input_ids": input_ids,
        "labels": labels,
        "num_samples": shard_samples,
        "num_tokens": shard_tokens,
        "num_chunks": len(input_ids),
        "block_size": block_size,
    }
    with open(shard_path, "wb") as f:
        pickle.dump(shard_data, f)
    log.info(
        f"Flushed shard {shard_id} with {len(input_ids)} chunks "
        f"({shard_tokens:,} tokens, {shard_samples} samples) -> {shard_path}"
    )
    return {
        "path": shard_filename,
        "num_samples": shard_samples,
        "num_tokens": shard_tokens,
        "num_chunks": len(input_ids),
    }


def _generate_cache_key_from_file(file_path, tokenizer, block_size):
    """Generate cache key based on file path, tokenizer, and block_size."""
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


def main():
    parser = argparse.ArgumentParser(description="Pretokenize large-scale datasets")
    
    # Model arguments
    parser.add_argument(
        "--input_model_filename",
        type=str,
        required=True,
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for tokenizer and pretokenized data"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data_local_path",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--eval_data_local_path",
        type=str,
        default=None,
        help="Path to evaluation JSONL file (optional)"
    )
    
    # Training arguments
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (block size)"
    )
    
    # Pretokenize specific arguments
    parser.add_argument(
        "--batch_size_samples",
        type=int,
        default=1000,
        help="Number of samples to process in each batch"
    )
    parser.add_argument(
        "--shard_size_tokens",
        type=int,
        default=1_000_000_000,
        help="Flush to disk after approximately this many tokens per shard"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of cache even if it exists"
    )
    parser.add_argument(
        "--estimate_memory",
        action="store_true",
        help="Estimate memory requirements and exit"
    )
    parser.add_argument(
        "--num_tokens_estimate",
        type=int,
        default=None,
        help="Estimated number of tokens for memory estimation (e.g., 100000000000 for 100B)"
    )
    
    args = parser.parse_args()
    
    # Memory estimation
    if args.estimate_memory:
        if args.num_tokens_estimate:
            estimates = estimate_memory_requirements(
                args.num_tokens_estimate, 
                args.model_max_length
            )
            log.info("=" * 60)
            log.info("Memory Requirements Estimation")
            log.info("=" * 60)
            log.info(f"Total tokens: {args.num_tokens_estimate:,}")
            log.info(f"Block size: {args.model_max_length}")
            log.info(f"Tokens memory (input_ids + labels): {estimates['tokens_memory_gb']:.2f} GB")
            log.info(f"Chunk overhead: {estimates['chunk_overhead_gb']:.2f} GB")
            log.info(f"Python overhead: {estimates['python_overhead_gb']:.2f} GB")
            log.info(f"Total estimated memory: {estimates['total_memory_gb']:.2f} GB")
            log.info(f"Recommended batch size: {estimates['recommended_batch_size_tokens']:,} tokens")
            log.info("=" * 60)
            log.info("Note: With streaming processing, actual memory usage will be much lower")
            log.info("      as data is processed in batches and saved incrementally.")
        else:
            log.error("--num_tokens_estimate is required when using --estimate_memory")
        return
    
    # Load tokenizer
    log.info("Loading tokenizer...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.input_model_filename,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    except Exception as e:
        log.warning(f"Failed to load fast tokenizer: {e}, trying slow tokenizer")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.input_model_filename,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log.info("Tokenizer loaded successfully")
    
    # Process training data
    if args.force:
        cache_path = os.path.join(args.cache_dir or ".", "pretokenized_*.pkl")
        manifest_path = os.path.join(args.cache_dir or ".", "pretokenized_*_manifest.json")
        import glob
        for path in glob.glob(cache_path):
            os.remove(path)
            log.info(f"Removed existing cache: {path}")
        for path in glob.glob(manifest_path):
            os.remove(path)
            log.info(f"Removed existing manifest: {path}")
    
    log.info("Processing training data...")
    train_cache = process_data_streaming(
        input_path=args.train_data_local_path,
        tokenizer=tokenizer,
        block_size=args.model_max_length,
        cache_dir=args.cache_dir or "./cache",
        batch_size_samples=args.batch_size_samples,
        shard_size_tokens=args.shard_size_tokens,
        max_samples=args.max_samples,
    )
    
    # Process evaluation data if provided
    if args.eval_data_local_path:
        log.info("Processing evaluation data...")
        eval_cache = process_data_streaming(
            input_path=args.eval_data_local_path,
            tokenizer=tokenizer,
            block_size=min(args.model_max_length, 1024),
            cache_dir=args.cache_dir or "./cache",
            batch_size_samples=args.batch_size_samples,
            shard_size_tokens=args.shard_size_tokens,
            max_samples=args.max_samples,
        )
        log.info(f"Evaluation cache: {eval_cache}")
    
    log.info(f"Pretokenization complete! Training cache: {train_cache}")


if __name__ == "__main__":
    main()
