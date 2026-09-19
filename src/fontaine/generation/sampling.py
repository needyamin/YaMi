"""Token sampling strategies: temperature, top-k, top-p, repetition penalty.

Pure functions over logits so behavior is unit-testable and the inference
engine stays a thin loop. Same functions power future server-side batching.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingConfig:
    """Decoding-time knobs (mirrors ``InferenceConfig``)."""

    temperature: float = 0.8
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    max_new_tokens: int = 256
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: list[int], penalty: float
) -> torch.Tensor:
    """Divide positive / multiply negative logits of already-generated tokens."""
    if penalty == 1.0 or not token_ids:
        return logits
    logits = logits.clone()
    unique = torch.tensor(list(set(token_ids)), dtype=torch.long, device=logits.device)
    values = logits[unique]
    logits[unique] = torch.where(values < 0, values * penalty, values / penalty)
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` highest-probability logits."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability reaches ``top_p``."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, stable=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Drop tokens once cumulative mass (excluding the current token) exceeds top_p;
    # always keep at least one token.
    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    remove[..., 0] = False
    mask = torch.zeros_like(remove, dtype=torch.bool).scatter_(dim=-1, index=sorted_idx, src=remove)
    return logits.masked_fill(mask, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generated_ids: list[int],
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from a logits vector of shape [vocab_size]."""
    config.validate()
    logits = apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

    if config.temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / config.temperature
    logits = apply_top_k(logits, config.top_k)
    logits = apply_top_p(logits, config.top_p)
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
