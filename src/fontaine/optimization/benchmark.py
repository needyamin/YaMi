"""Small forward/backward timings. Large configs are refused."""

import time

import torch

from fontaine.config.schema import ModelConfig
from fontaine.models import build_model
from fontaine.optimization.memory import estimate_parameter_count


def run_benchmarks(config: ModelConfig, allow_large: bool = False, steps: int = 2) -> dict[str, float]:
    counts = estimate_parameter_count(config)
    if counts["total"] > 5_000_000 and not allow_large:
        raise RuntimeError(
            f"this config has {counts['total']:,} parameters. "
            "benchmark allocates the model. Pass --allow-large to continue, "
            "or select a tiny profile."
        )
    model = build_model(config)
    model.train()
    tokens = torch.randint(0, int(config.vocab_size), (2, min(32, config.max_sequence_length - 1)))
    targets = torch.randint(0, int(config.vocab_size), tokens.shape)
    start = time.perf_counter()
    for _ in range(steps):
        output = model(tokens, targets=targets)
        output.loss.backward()
        model.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - start
    tokens_per_step = tokens.numel()
    return {
        "parameters": float(counts["total"]),
        "active_parameters": float(counts["active"]),
        "steps": float(steps),
        "seconds": elapsed,
        "tokens_per_second": tokens_per_step * steps / elapsed,
    }
