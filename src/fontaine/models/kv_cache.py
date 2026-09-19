"""KV cache for incremental (token-by-token) generation.

The cache is a single pre-allocated tensor per K and V, sized to
``model.max_sequence_length``. Pre-allocation avoids per-step tensor
concatenation (O(T^2) copies) and mirrors how production serving stacks
allocate; a paged/ dynamic allocator is the future upgrade path
(see ``docs/inference/architecture.md``).
"""

from dataclasses import dataclass

import torch

from fontaine.config.schema import ModelConfig


@dataclass
class KVCache:
    """Pre-allocated per-layer key/value store.

    ``update`` writes the new K/V slice for one layer at the current position
    and returns the full K/V (past + present) for that layer. The position is
    advanced once per decode step by the caller via :meth:`advance`, after all
    layers have been updated.
    """

    keys: torch.Tensor  # [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
    values: torch.Tensor  # same shape
    pos: int = 0

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        shape = (
            config.num_layers,
            batch_size,
            config.num_kv_heads,
            config.max_sequence_length,
            config.head_dim,
        )
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
        )

    @property
    def max_seq_len(self) -> int:
        return self.keys.shape[3]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one layer's K/V for ``T`` new positions and return views of all K/V.

        ``k``/``v`` have shape ``[batch, num_kv_heads, T, head_dim]``.
        """
        t = k.shape[2]
        end = self.pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: positions [{self.pos}, {end}) exceed "
                f"max_sequence_length={self.max_seq_len}"
            )
        self.keys[layer_idx, :, :, self.pos : end] = k
        self.values[layer_idx, :, :, self.pos : end] = v
        return self.keys[layer_idx, :, :, :end], self.values[layer_idx, :, :, :end]

    def advance(self, t: int) -> None:
        """Advance the write position after a forward pass wrote ``t`` tokens."""
        self.pos += t

    def reset(self) -> None:
        """Clear the cache (reusing the allocation) for a new generation."""
        self.pos = 0
