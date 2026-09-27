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

    def allocated_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()


class DynamicKVCache:
    """KV cache that grows with the sequence instead of reserving the full window."""

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.pos = 0
        self.keys = torch.empty(
            config.num_layers, batch_size, config.num_kv_heads, 0, config.head_dim,
            device=device, dtype=dtype,
        )
        self.values = torch.empty_like(self.keys)

    @property
    def max_seq_len(self) -> int:
        return self.config.max_sequence_length

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        end = self.pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: positions [{self.pos}, {end}) exceed "
                f"max_sequence_length={self.max_seq_len}"
            )
        self._grow(end)
        self.keys[layer_idx, :, :, self.pos : end] = k
        self.values[layer_idx, :, :, self.pos : end] = v
        return self.keys[layer_idx, :, :, :end], self.values[layer_idx, :, :, :end]

    def _grow(self, needed: int) -> None:
        current = self.keys.shape[3]
        if needed <= current:
            return
        size = max(needed, max(16, current * 2))
        size = min(size, self.max_seq_len)
        shape = (
            self.config.num_layers,
            self.batch_size,
            self.config.num_kv_heads,
            size,
            self.config.head_dim,
        )
        keys = torch.zeros(shape, device=self.device, dtype=self.dtype)
        values = torch.zeros_like(keys)
        if current:
            keys[:, :, :, :current] = self.keys
            values[:, :, :, :current] = self.values
        self.keys = keys
        self.values = values

    def advance(self, t: int) -> None:
        self.pos += t

    def reset(self) -> None:
        self.pos = 0

    def allocated_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()


class PagedKVCache:
    """Block-table KV cache. Pages are allocated as tokens arrive."""

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        page_size: int = 16,
    ) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.page_size = page_size
        self.pos = 0
        self.pages_k: list[torch.Tensor] = []
        self.pages_v: list[torch.Tensor] = []

    @property
    def max_seq_len(self) -> int:
        return self.config.max_sequence_length

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        end = self.pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: positions [{self.pos}, {end}) exceed "
                f"max_sequence_length={self.max_seq_len}"
            )
        self._ensure_pages(end)
        self._write(self.pages_k, layer_idx, self.pos, k)
        self._write(self.pages_v, layer_idx, self.pos, v)
        return self._gather(self.pages_k, layer_idx, end), self._gather(self.pages_v, layer_idx, end)

    def _ensure_pages(self, needed: int) -> None:
        pages_needed = (needed + self.page_size - 1) // self.page_size
        while len(self.pages_k) < pages_needed:
            shape = (
                self.config.num_layers,
                self.batch_size,
                self.config.num_kv_heads,
                self.page_size,
                self.config.head_dim,
            )
            self.pages_k.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
            self.pages_v.append(torch.zeros(shape, device=self.device, dtype=self.dtype))

    def _write(self, pages: list[torch.Tensor], layer_idx: int, start: int, values: torch.Tensor) -> None:
        offset = 0
        remaining = values.shape[2]
        pos = start
        while remaining:
            page_index = pos // self.page_size
            page_offset = pos % self.page_size
            take = min(remaining, self.page_size - page_offset)
            pages[page_index][layer_idx, :, :, page_offset : page_offset + take] = values[:, :, offset : offset + take]
            offset += take
            pos += take
            remaining -= take

    def _gather(self, pages: list[torch.Tensor], layer_idx: int, end: int) -> torch.Tensor:
        if end == 0:
            return pages[0][layer_idx, :, :, :0]
        chunks = []
        pos = 0
        while pos < end:
            page_index = pos // self.page_size
            page_offset = pos % self.page_size
            take = min(end - pos, self.page_size - page_offset)
            chunks.append(pages[page_index][layer_idx, :, :, page_offset : page_offset + take])
            pos += take
        return torch.cat(chunks, dim=2)

    def advance(self, t: int) -> None:
        self.pos += t

    def reset(self) -> None:
        self.pos = 0
        self.pages_k.clear()
        self.pages_v.clear()

    def allocated_bytes(self) -> int:
        total = 0
        for page in self.pages_k + self.pages_v:
            total += page.numel() * page.element_size()
        return total


def build_kv_cache(
    config: ModelConfig,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    kind: str = "static",
):
    """Allocate a static, dynamic, or paged cache. Unknown kinds fail."""
    if kind == "static":
        return KVCache.from_config(config, batch_size, device, dtype)
    if kind == "dynamic":
        return DynamicKVCache(config, batch_size, device, dtype)
    if kind == "paged":
        return PagedKVCache(config, batch_size, device, dtype)
    raise ValueError(
        f"unknown kv cache {kind!r}. Choose static, dynamic, or paged."
    )
