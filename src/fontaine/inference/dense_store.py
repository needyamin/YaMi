"""Stream dense transformer blocks from disk so a big model fits a RAM floor.

Serving concept: keep a floor of dense (active) block weight resident in
memory — 12 GiB by default on the web-chat path — and let the remaining block
weight live in storage. A forward pass materializes each block on demand (LRU
hit, or a disk load admitted into the budget) and prefetches the next block on
a background thread so the disk read overlaps the current block's compute.

Whole blocks are pickled to disk and replaced by lightweight placeholders, so
the mechanism is agnostic to what the blocks contain — fp32 parameters, int8
dynamic-quantized linears, custom kernels — and restoring is exact: a block
reloads as the very module that was spilled, byte for byte. Inference only; a
placeholder block has no weights to train.

The mixture-of-experts store (``fontaine.inference.expert_store``) streams
routed experts with the same lease discipline; the two compose: this store
treats each block as one unit, the expert store serves individual experts
inside a resident block.
"""

from __future__ import annotations

import queue
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from fontaine.utils.logging import get_logger

logger = get_logger("inference.dense")

# Web-chat serving default: at least this much dense block weight stays
# resident in RAM; the rest streams from disk. 12288 MiB = 12 GiB.
DEFAULT_DENSE_BUDGET_MB = 12288

_MIB = 1024 * 1024


@dataclass
class DenseStoreStats:
    """Counters for one streaming block list."""

    requests: int = 0
    hits: int = 0
    misses: int = 0
    prefetched: int = 0
    bytes_read: int = 0
    resident_bytes: int = 0
    capacity_bytes: int = 0


def _state_nbytes(state: dict) -> int:
    """Resident bytes for one block's tensors."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )


class _StreamedBlock(nn.Module):
    """Placeholder left in the module tree while a block lives on disk."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.block_index = index

    def forward(self, *args, **kwargs):  # pragma: no cover - store materializes first
        raise RuntimeError(
            f"block {self.block_index} is streamed from disk but was not materialized; "
            "run the model through its DenseStreamList"
        )


class DenseStreamList(nn.ModuleList):
    """An ``nn.ModuleList`` whose blocks can live on disk behind an LRU budget.

    ``nn.ModuleList.__iter__`` routes through ``__getitem__``, so the model's
    plain ``for block in self.blocks:`` transparently materializes each block.
    Entries swap between the real module (resident) and a ``_StreamedBlock``
    placeholder (on disk). Evicting a block that is mid-forward is safe: the
    running code holds its own reference, and the ModuleList entry is only
    consulted again on the next call.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        budget_bytes: int,
        directory: str | Path,
        *,
        overlap: bool = True,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be > 0, got {budget_bytes}")
        if not blocks:
            raise ValueError("blocks must contain at least one transformer block")
        super().__init__(blocks)
        self.capacity_bytes = int(budget_bytes)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Plain (non-registered) containers: placeholders and cached modules
        # must not re-register as children of this list.
        self._placeholders: dict[int, _StreamedBlock] = {}
        self._resident: OrderedDict[int, int] = OrderedDict()  # index -> nbytes
        self._modules_cache: dict[int, nn.Module] = {}
        self._stats = DenseStoreStats(capacity_bytes=self.capacity_bytes)
        self._lock = threading.RLock()
        self._queue: queue.Queue[int | None] | None = None
        self._thread: threading.Thread | None = None
        if overlap:
            self._queue = queue.Queue()
            self._thread = threading.Thread(
                target=self._worker, name="yami-dense-prefetch", daemon=True
            )
            self._thread.start()

    # -- spill ------------------------------------------------------------------

    def spill_all(self) -> None:
        """Pickle every block to disk and replace it with a placeholder."""
        for index in range(len(self)):
            real = super().__getitem__(index)
            torch.save(real, self._path(index))
            placeholder = _StreamedBlock(index)
            self._placeholders[index] = placeholder
            setattr(self, str(index), placeholder)  # ModuleList.__setitem__ semantics
            del real
        logger.info(
            "dense streaming blocks=%d budget_bytes=%d directory=%s",
            len(self),
            self.capacity_bytes,
            self.directory,
        )

    def _path(self, index: int) -> Path:
        return self.directory / f"block_{index:04d}.pt"

    # -- module-list protocol ------------------------------------------------------

    def __getitem__(self, idx):  # type: ignore[override]
        if isinstance(idx, slice):
            # A sliced copy is materialized eagerly; streaming applies to the
            # sequential forward, not to copies.
            return nn.ModuleList([self[i] for i in range(*idx.indices(len(self)))])
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"block index must be an integer, got {type(idx)!r}")
        count = len(self)
        index = idx + count if idx < 0 else idx
        if not 0 <= index < count:
            raise IndexError(f"block index {idx} out of range for {count} blocks")
        module = self._ensure_resident(index)
        self._maybe_prefetch(index + 1)
        if module is not None:
            return module
        return super().__getitem__(index)

    def __iter__(self):  # type: ignore[override]
        for index in range(len(self)):
            yield self[index]

    # -- cache ------------------------------------------------------------------

    def _ensure_resident(self, index: int) -> nn.Module | None:
        """Return the real module for ``index``, materializing it if needed.

        Returns ``None`` when the entry is not streamed (attach skipped it or
        the store is a passthrough), which callers serve as-is.
        """
        with self._lock:
            self._stats.requests += 1
            if index in self._resident:
                self._stats.hits += 1
                self._resident.move_to_end(index)
                return self._modules_cache[index]
            if index not in self._placeholders:
                return None  # not streamed
            self._stats.misses += 1
        module = self._load(index)
        with self._lock:
            # Another thread (prefetch) may have admitted it meanwhile.
            if index in self._resident:
                self._stats.hits += 1
                self._resident.move_to_end(index)
                return self._modules_cache[index]
            nbytes = _state_nbytes(module.state_dict())
            self._stats.bytes_read += nbytes
            self._admit(index, module, nbytes)
            return self._modules_cache[index]

    def _admit(self, index: int, module: nn.Module, nbytes: int) -> None:
        """Insert a materialized block, evicting cold entries to stay in budget.

        Called with the lock held. Evicting the block that is mid-forward is
        safe — its execution holds a live reference — but it costs a reload,
        so the LRU order (most recent last) keeps that rare.
        """
        while self._resident and sum(self._resident.values()) + nbytes > self.capacity_bytes:
            victim, victim_bytes = next(iter(self._resident.items()))
            self._evict(victim, victim_bytes)
        self._modules_cache[index] = module
        if sum(self._resident.values()) + nbytes <= self.capacity_bytes:
            self._resident[index] = nbytes
            self._resident.move_to_end(index)
            setattr(self, str(index), module)
        else:
            # A single block larger than the whole budget: serve it straight
            # from the transient cache entry without claiming budget space.
            setattr(self, str(index), module)

    def _evict(self, index: int, nbytes: int) -> None:
        self._resident.pop(index, None)
        self._modules_cache.pop(index, None)
        placeholder = self._placeholders.get(index)
        if placeholder is not None:
            setattr(self, str(index), placeholder)

    def _load(self, index: int) -> nn.Module:
        path = self._path(index)
        if not path.is_file():
            raise FileNotFoundError(f"missing streamed block at {path}")
        # Materialization can happen inside the engine's inference_mode; load
        # with inference mode off so cached tensors stay ordinary tensors that
        # later requests (and the prefetch thread) can reuse freely. The file
        # was written by this process (attach), so unpickling our own module
        # classes with weights_only=False is trusted.
        with torch.inference_mode(False), torch.no_grad():
            module = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(module, nn.Module):
            raise RuntimeError(f"streamed block {path} is not an nn.Module")
        return module

    # -- prefetch -----------------------------------------------------------------

    def _maybe_prefetch(self, index: int) -> None:
        """Warm ``index`` on the worker thread while the caller computes."""
        if self._queue is None or not 0 <= index < len(self):
            return
        with self._lock:
            known = index in self._resident or index not in self._placeholders
        if not known:
            self._queue.put(index)

    def prefetch(self, index: int) -> None:
        """Advise the cache to load ``index`` ahead of its forward pass."""
        self._maybe_prefetch(index)

    def _worker(self) -> None:
        assert self._queue is not None
        while True:
            index = self._queue.get()
            if index is None:
                return
            try:
                with self._lock:
                    if index in self._resident:
                        self._stats.prefetched += 1
                        continue
                module = self._load(index)
                with self._lock:
                    if index in self._resident:
                        self._stats.prefetched += 1
                        continue
                    nbytes = _state_nbytes(module.state_dict())
                    self._stats.bytes_read += nbytes
                    self._stats.prefetched += 1
                    self._admit(index, module, nbytes)
            except Exception:
                logger.exception("dense prefetch failed for block %d", index)

    def advance_prefetch(self, index: int) -> None:
        """Called by the engine: warm ``index`` while ``index - 1`` computes."""
        self.prefetch(index)

    # -- reporting -------------------------------------------------------------------

    def stats(self) -> DenseStoreStats:
        """Return a snapshot of the counters."""
        with self._lock:
            return DenseStoreStats(
                requests=self._stats.requests,
                hits=self._stats.hits,
                misses=self._stats.misses,
                prefetched=self._stats.prefetched,
                bytes_read=self._stats.bytes_read,
                resident_bytes=sum(self._resident.values()),
                capacity_bytes=self.capacity_bytes,
            )

    def close(self) -> None:
        """Stop the prefetch thread."""
        if self._queue is None or self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._queue = None
        self._thread = None


def _blocks_bytes(blocks: list[nn.Module]) -> int:
    return sum(_state_nbytes(block.state_dict()) for block in blocks)


def attach_dense_streaming(
    model: nn.Module,
    budget_bytes: int,
    directory: str | Path | None = None,
    *,
    overlap: bool = True,
) -> DenseStreamList | None:
    """Spill ``model.blocks`` to disk and serve them from a RAM budget.

    Returns ``None`` when the model has no block list, the blocks already fit
    the budget (small models stay exactly as they were), or a non-CPU device
    makes streaming pointless. Call after precision conversion so the disk
    holds the served weight format (int8 on CPU), and never on a training
    path: a placeholder block cannot backprop.
    """
    blocks = list(getattr(model, "blocks", None) or [])
    if not blocks:
        return None
    total = _blocks_bytes(blocks)
    if total <= budget_bytes:
        logger.info(
            "dense streaming skipped: %d bytes of block weight fit the %d byte budget",
            total,
            budget_bytes,
        )
        return None
    spill_dir = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="yami-dense-"))
    stream_list = DenseStreamList(blocks, budget_bytes, spill_dir, overlap=overlap)
    stream_list.spill_all()
    setattr(model, "blocks", stream_list)
    return stream_list
