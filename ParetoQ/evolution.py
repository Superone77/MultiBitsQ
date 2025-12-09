import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.utils_quant import QuantizeLinear
from utils import datautils, utils


log = utils.get_logger("evolution")


def _parse_bits(bits: str) -> List[int]:
    return [int(x.strip()) for x in bits.split(",") if x.strip()]


def _resolve_dtype(dtype: str):
    name = dtype.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


@dataclass(frozen=True)
class Individual:
    bits: Tuple[int, ...]
    perplexity: float
    avg_bits: float
    total_bits: float

    def to_json(self, layer_names: Sequence[str]) -> Dict[str, object]:
        assignment = {name: bit for name, bit in zip(layer_names, self.bits)}
        return {
            "assignment": assignment,
            "perplexity": self.perplexity,
            "avg_bits": self.avg_bits,
            "total_bits": self.total_bits,
        }


def build_eval_loader(
    eval_path: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    max_eval_samples: int,
    num_proc: Optional[int] = None,
) -> DataLoader:
    """Tokenize evaluation set and return a PyTorch dataloader."""
    raw = load_dataset("json", data_files={"validation": eval_path})["validation"]
    tokenized = datautils._tokenize_and_group(
        raw, tokenizer, block_size=block_size, num_proc=num_proc, desc_prefix="eval"
    )
    if max_eval_samples and max_eval_samples > 0:
        tokenized = tokenized.select(range(min(max_eval_samples, len(tokenized))))

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate)


def collect_quant_layers(
    model: torch.nn.Module, candidate_bits: Optional[Iterable[int]] = None
) -> List[Tuple[str, torch.nn.Module, List[int], int]]:
    """
    Return all quantizable layers with their candidate bits and parameter counts.
    Format: (name, module, allowed_bits, num_params)
    """
    layers = []
    candidate_set = set(candidate_bits) if candidate_bits is not None else None

    for name, module in model.named_modules():
        has_bits = hasattr(module, "w_bits_list") and hasattr(module, "set_bits")
        if not has_bits:
            continue

        allowed = list(getattr(module, "w_bits_list", []))
        if candidate_set is not None:
            allowed = [b for b in allowed if b in candidate_set]
        if len(allowed) == 0:
            continue

        # Disable randomness during search to keep fitness deterministic.
        if hasattr(module, "multiple_bits_random_assign"):
            module.multiple_bits_random_assign = False
        if hasattr(module, "noise_injection"):
            module.noise_injection = False

        num_params = module.weight.numel() if hasattr(module, "weight") else 0
        layers.append((name, module, allowed, num_params))

    if len(layers) == 0:
        raise ValueError("No quantized layers with w_bits_list found in the loaded model.")
    return layers


def apply_bits(bits: Sequence[int], layers: List[Tuple[str, torch.nn.Module, List[int], int]]):
    """Set per-layer bit widths on the model in-place."""
    for bit, (_, module, _, _) in zip(bits, layers):
        module.set_bits(int(bit))


def compute_cost(bits: Sequence[int], layers: List[Tuple[str, torch.nn.Module, List[int], int]]) -> Tuple[float, float]:
    total_params = sum(layer[3] for layer in layers)
    total_bits = sum(bit * layer[3] for bit, layer in zip(bits, layers))
    avg_bits = total_bits / max(total_params, 1)
    return avg_bits, total_bits


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Return perplexity on the provided dataloader."""
    model.eval()
    losses: List[float] = []
    amp_ctx = torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_ctx():
                outputs = model(**batch)
                loss = outputs.loss
            losses.append(loss.detach().float().item())

    mean_loss = sum(losses) / max(len(losses), 1)
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def sample_individual(
    layers: List[Tuple[str, torch.nn.Module, List[int], int]],
    max_avg_bits: Optional[float],
    max_bit_budget: Optional[float],
    attempts: int = 50,
) -> Optional[Tuple[int, ...]]:
    """Draw a random bit assignment that satisfies constraints."""
    for _ in range(attempts):
        candidate = [random.choice(layer[2]) for layer in layers]
        avg_bits, total_bits = compute_cost(candidate, layers)
        if (max_avg_bits is None or avg_bits <= max_avg_bits) and (
            max_bit_budget is None or total_bits <= max_bit_budget
        ):
            return tuple(candidate)
    return None


def mutate(
    bits: Sequence[int],
    layers: List[Tuple[str, torch.nn.Module, List[int], int]],
    mutation_prob: float,
) -> Tuple[int, ...]:
    mutated = list(bits)
    for idx, (_, _, options, _) in enumerate(layers):
        if random.random() < mutation_prob:
            choices = [b for b in options if b != mutated[idx]]
            if len(choices) > 0:
                mutated[idx] = random.choice(choices)
    return tuple(mutated)


def crossover(parent_a: Sequence[int], parent_b: Sequence[int]) -> Tuple[int, ...]:
    return tuple(random.choice([a, b]) for a, b in zip(parent_a, parent_b))


def evolutionary_search(
    model: torch.nn.Module,
    layers: List[Tuple[str, torch.nn.Module, List[int], int]],
    dataloader: DataLoader,
    device: torch.device,
    population_size: int,
    generations: int,
    top_k: int,
    mutation_prob: float,
    crossover_prob: float,
    max_avg_bits: Optional[float],
    max_bit_budget: Optional[float],
) -> Individual:
    cache: Dict[Tuple[int, ...], Individual] = {}
    population: List[Individual] = []

    def evaluate(bits: Tuple[int, ...]) -> Individual:
        if bits in cache:
            return cache[bits]
        apply_bits(bits, layers)
        perplexity = evaluate_model(model, dataloader, device)
        avg_bits, total_bits = compute_cost(bits, layers)
        indiv = Individual(bits=bits, perplexity=perplexity, avg_bits=avg_bits, total_bits=total_bits)
        cache[bits] = indiv
        return indiv

    # Initialize population
    while len(population) < population_size:
        sampled = sample_individual(layers, max_avg_bits, max_bit_budget)
        if sampled is None:
            raise RuntimeError("Failed to sample a valid initial population under the given constraints.")
        population.append(evaluate(sampled))
        log.info("Init %d/%d: ppl=%.3f avg_bits=%.3f", len(population), population_size, population[-1].perplexity, population[-1].avg_bits)

    for gen in range(generations):
        population.sort(key=lambda x: x.perplexity)
        best = population[0]
        log.info(
            "Gen %d best perplexity=%.3f avg_bits=%.3f",
            gen,
            best.perplexity,
            best.avg_bits,
        )

        parents = population[: max(1, min(top_k, len(population)))]
        next_population = parents.copy()

        while len(next_population) < population_size:
            parent_a, parent_b = random.sample(parents, k=2) if len(parents) >= 2 else (parents[0], parents[0])
            child_bits = crossover(parent_a.bits, parent_b.bits) if random.random() < crossover_prob else parent_a.bits
            child_bits = mutate(child_bits, layers, mutation_prob)
            avg_bits, total_bits = compute_cost(child_bits, layers)
            if (max_avg_bits is not None and avg_bits > max_avg_bits) or (
                max_bit_budget is not None and total_bits > max_bit_budget
            ):
                continue
            next_population.append(evaluate(child_bits))

        population = next_population

    population.sort(key=lambda x: x.perplexity)
    return population[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary search for mixed-precision quantization.")
    parser.add_argument("--model_path", required=True, help="Path to the trained multi-bit checkpoint.")
    parser.add_argument("--eval_data", required=True, help="Path to jsonl file with a 'text' field for evaluation.")
    parser.add_argument("--candidate_bits", type=str, default="2,3,4", help="Comma separated candidate bit widths.")
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=4, help="Number of top individuals kept each generation.")
    parser.add_argument("--mutation_prob", type=float, default=0.1)
    parser.add_argument("--crossover_prob", type=float, default=0.5)
    parser.add_argument("--max_avg_bits", type=float, default=None, help="Optional constraint on average bit-width.")
    parser.add_argument("--max_bit_budget", type=float, default=None, help="Optional total bit budget (params*bits).")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_eval_samples", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=None, help="Parallel workers for tokenization.")
    parser.add_argument("--dtype", type=str, default="bf16", help="Computation dtype: fp16, bf16, or fp32.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_result_path", type=str, default=None, help="Optional path to dump the best assignment json.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    candidate_bits = _parse_bits(args.candidate_bits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype)

    log.info("Loading tokenizer and model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_pretrained(args.model_path)
    config.w_bits_list = candidate_bits
    config.noise_injection = False
    model = LlamaForCausalLMQuant.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)

    layers = collect_quant_layers(model, candidate_bits)
    log.info("Found %d quantized layers to search over.", len(layers))

    eval_loader = build_eval_loader(
        eval_path=args.eval_data,
        tokenizer=tokenizer,
        block_size=args.block_size,
        batch_size=args.eval_batch_size,
        max_eval_samples=args.max_eval_samples,
        num_proc=args.num_proc,
    )

    best = evolutionary_search(
        model=model,
        layers=layers,
        dataloader=eval_loader,
        device=device,
        population_size=args.population_size,
        generations=args.generations,
        top_k=args.top_k,
        mutation_prob=args.mutation_prob,
        crossover_prob=args.crossover_prob,
        max_avg_bits=args.max_avg_bits,
        max_bit_budget=args.max_bit_budget,
    )

    layer_names = [name for name, _, _, _ in layers]
    log.info(
        "Best assignment: ppl=%.3f avg_bits=%.3f",
        best.perplexity,
        best.avg_bits,
    )
    for name, bit in zip(layer_names, best.bits):
        log.info("  %s -> %dbit", name, bit)

    if args.save_result_path:
        save_dir = os.path.dirname(args.save_result_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save_result_path, "w") as f:
            json.dump(best.to_json(layer_names), f, indent=2)
        log.info("Saved best assignment to %s", args.save_result_path)


if __name__ == "__main__":
    main()
