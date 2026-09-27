"""Run a long sequence in chunks, carrying a KV cache between chunks."""

import torch

from fontaine.models.kv_cache import build_kv_cache
from fontaine.models.transformer import FontaineModel


def forward_chunked(
    model: FontaineModel, input_ids: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Return logits for ``input_ids`` without feeding the whole sequence at once.

    Full-attention layers match a single forward. The cache is dynamic, so the
    call does not reserve ``max_sequence_length`` up front.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    batch, length = input_ids.shape
    cache = build_kv_cache(model.config, batch, input_ids.device, torch.float32, "dynamic")
    pieces = []
    for start in range(0, length, chunk_size):
        chunk = input_ids[:, start : start + chunk_size]
        logits = model(chunk, cache=cache).logits
        cache.advance(chunk.shape[1])
        pieces.append(logits)
    return torch.cat(pieces, dim=1)
