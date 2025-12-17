import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import transformers

from models.configuration_llama import LlamaConfig
from utils import datautils


log = logging.getLogger("datautils_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a jsonl dataset and ensure token ids stay within the model vocabulary."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the base model (same as input_model_filename in training).",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the jsonl dataset to stream.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Block size used for packing tokens (should match model_max_length).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for tokenizer.map (matches StreamingJsonlDataset default in training).",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Optional cap on yielded sequences for a quick dry run. 0 means full pass.",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=100000,
        help="Token interval for progress logging.",
    )
    parser.add_argument(
        "--dataset_label",
        type=str,
        default="train",
        help="Label for logging context (e.g., train/eval).",
    )
    return parser.parse_args()


def load_vocab_size(model_path: str, tokenizer: transformers.PreTrainedTokenizerBase) -> int:
    config = LlamaConfig.from_pretrained(model_path)
    vocab_size = getattr(config, "vocab_size", None) or tokenizer.vocab_size
    if vocab_size is None:
        raise ValueError("Failed to determine vocab size from config or tokenizer.")
    return int(vocab_size)


def log_progress_if_needed(dataset: datautils.StreamingJsonlDataset, next_report: int, print_interval: int) -> int:
    if dataset.total_yielded_tokens >= next_report:
        log.info(
            "Progress: yielded_tokens=%d processed_tokens=%d",
            dataset.total_yielded_tokens,
            dataset.total_processed_tokens,
        )
        next_report += print_interval
    return next_report


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    data_path = Path(args.data_path)
    if not data_path.exists():
        log.error("Dataset not found: %s", data_path)
        return 1

    log.info("Loading tokenizer from %s", args.model_path)
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
    vocab_size = load_vocab_size(args.model_path, tokenizer)
    log.info("Vocab size: %d", vocab_size)

    dataset = datautils.StreamingJsonlDataset(
        path=str(data_path),
        tokenizer=tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    overflow_examples: list[dict[str, Optional[int]]] = []
    max_token_id = -1
    min_token_id = None
    total_sequences = 0
    total_tokens_checked = 0
    next_report = args.print_interval

    for batch in dataset:
        seq_tensor = batch["input_ids"]
        batch_max = int(torch.max(seq_tensor).item())
        batch_min = int(torch.min(seq_tensor).item())
        max_token_id = max(max_token_id, batch_max)
        min_token_id = batch_min if min_token_id is None else min(min_token_id, batch_min)

        total_sequences += 1
        total_tokens_checked += seq_tensor.numel()

        if batch_max >= vocab_size or batch_min < 0:
            overflow_examples.append(
                {
                    "sequence_index": total_sequences,
                    "max_id": batch_max,
                    "min_id": batch_min,
                }
            )
            if len(overflow_examples) == 1:
                decoded_preview = tokenizer.decode(seq_tensor[: min(32, seq_tensor.numel())])
                log.error(
                    "Out-of-vocab tokens detected in sequence %d (min=%d, max=%d). Preview: %s",
                    total_sequences,
                    batch_min,
                    batch_max,
                    decoded_preview,
                )

        if args.max_sequences > 0 and total_sequences >= args.max_sequences:
            log.info("Reached max_sequences=%d, stopping early.", args.max_sequences)
            break

        next_report = log_progress_if_needed(dataset, next_report, args.print_interval)

    stats = dataset.get_token_counts()
    log.info(
        "[%s] Sequences=%d tokens_checked=%d processed_tokens=%d yielded_tokens=%d max_token_id=%d min_token_id=%s",
        args.dataset_label,
        total_sequences,
        total_tokens_checked,
        stats["total_processed_tokens"],
        stats["total_yielded_tokens"],
        max_token_id,
        "n/a" if min_token_id is None else min_token_id,
    )

    if overflow_examples:
        log.error(
            "Found %d sequence(s) with ids outside vocab_size=%d. First occurrence at sequence %d (min=%d, max=%d).",
            len(overflow_examples),
            vocab_size,
            overflow_examples[0]["sequence_index"],
            overflow_examples[0]["min_id"],
            overflow_examples[0]["max_id"],
        )
        return 2

    log.info("All checked sequences stay within vocab size %d.", vocab_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
