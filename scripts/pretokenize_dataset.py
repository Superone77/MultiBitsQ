import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import transformers

# Ensure we can import ParetoQ.utils.*
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ParetoQ"))

from utils import datautils  # noqa: E402

log = logging.getLogger("pretokenize")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_tokenizer(path: str, cache_dir: str | None, use_fast: bool, max_len: int):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            cache_dir=cache_dir,
            model_max_length=max_len,
            padding_side="right",
            use_fast=use_fast,
        )
        log.info("Loaded %s tokenizer successfully", "fast" if use_fast else "slow")
        return tokenizer
    except Exception as exc:  # noqa: BLE001
        if use_fast:
            log.warning("Fast tokenizer failed (%s), falling back to slow.", exc)
            return load_tokenizer(path, cache_dir, use_fast=False, max_len=max_len)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize jsonl data into cached HuggingFace datasets for reuse."
    )
    parser.add_argument(
        "--train_path", required=True, help="Path to training jsonl with a 'text' field per line."
    )
    parser.add_argument(
        "--eval_path", default=None, help="Optional eval/validation jsonl with a 'text' field per line."
    )
    parser.add_argument(
        "--tokenizer_path", required=True, help="Tokenizer/model path for AutoTokenizer.from_pretrained."
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Directory to store the tokenized cache. Default: alongside train file.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Sequence length to pack into. Must align with training model_max_length.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Optional limit for training samples after packing (mainly for smoke tests).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="Optional limit for eval samples after packing.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Parallel workers for tokenization; overrides TOKENIZER_NUM_PROC if set.",
    )
    parser.add_argument(
        "--use_fast",
        action="store_true",
        help="Try loading the fast tokenizer first (recommended if supported).",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(
        args.tokenizer_path, args.cache_dir, use_fast=args.use_fast, max_len=args.block_size
    )

    env_num_proc = os.getenv("TOKENIZER_NUM_PROC")
    if args.num_proc is None and env_num_proc:
        try:
            args.num_proc = int(env_num_proc)
        except ValueError:
            log.warning("Ignored TOKENIZER_NUM_PROC=%s (not an int)", env_num_proc)

    train_ds, eval_ds = datautils.prepare_tokenized_datasets(
        train_path=args.train_path,
        valid_path=args.eval_path,
        tokenizer=tokenizer,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        rank=0,
        world_size=1,
    )

    train_len = datautils.get_dataset_length(train_ds)
    eval_len = datautils.get_dataset_length(eval_ds) if eval_ds is not None else None

    log.info("Pre-tokenization complete.")
    if train_len is not None:
        log.info("Train samples cached: %s", train_len)
    if eval_len is not None:
        log.info("Eval samples cached: %s", eval_len)


if __name__ == "__main__":
    main()
