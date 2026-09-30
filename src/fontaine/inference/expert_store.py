"""Stream routed experts from disk so a huge mixture-of-experts model fits in RAM.

The dense trunk stays resident: token embeddings, attention, norms, the router,
and any shared experts. Routed experts are files. A forward loads only the
experts the router selected, keeps a per-layer hot set inside a byte budget,
and prefetches the next layer's previous choices so the read can overlap the
next attention block.

A lookup holds a lease until release. Prefetch is advisory: it fills free
space and will not evict a leased expert, and a colder expert will not replace
a hotter one without the usual promotion margin. When the budget cannot hold
the expert, the weights live only for that call and are dropped on release.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fontaine.models.components import MixtureOfExperts
from fontaine.models.kernels import linear, swiglu
from fontaine.optimization.expert_tier import tier_decay_value, tier_lfru_score
from fontaine.utils.logging import get_logger

logger = get_logger("inference.experts")

_DECAY_EVERY = 64
_U32 = 0xFFFFFFFF


@dataclass
class ExpertStoreStats:
    """Counters for one streaming store."""

    requests: int = 0
    hits: int = 0
    misses: int = 0
    prefetched: int = 0
    prefetch_hits: int = 0
    bytes_read: int = 0
    resident_bytes: int = 0
    capacity_bytes: int = 0


@dataclass
class _Slot:
    layer: int
    expert: int
    weights: dict[str, torch.Tensor]
    nbytes: int
    leases: int = 0


@dataclass
class _View:
    layer: int
    expert: int
    weights: dict[str, torch.Tensor]
    slot: _Slot | None
    released: bool = False


def _nbytes(weights: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in weights.values())


def feedforward_from_state(
    state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    activation: str,
    kernel: str,
) -> torch.Tensor:
    """Run one expert from a state dict on ``inputs``."""

    def project(prefix: str, hidden: torch.Tensor) -> torch.Tensor:
        weight = state[f"{prefix}.weight"]
        if weight.device != hidden.device or weight.dtype != hidden.dtype:
            weight = weight.to(device=hidden.device, dtype=hidden.dtype)
        bias = state.get(f"{prefix}.bias")
        if bias is not None and (bias.device != hidden.device or bias.dtype != hidden.dtype):
            bias = bias.to(device=hidden.device, dtype=hidden.dtype)
        if kernel == "reference":
            return linear(hidden, weight, bias, "reference")
        return F.linear(hidden, weight, bias)

    if activation == "swiglu":
        gate = project("gate_proj", inputs)
        up = project("up_proj", inputs)
        return project("down_proj", swiglu(gate, up, kernel))
    if activation == "gelu":
        hidden = F.gelu(project("fc", inputs))
        return project("proj", hidden)
    raise ValueError(f"unknown activation: {activation!r}")


class ExpertStore:
    """Disk-backed expert weights with a leased RAM cache.

    ``budget_bytes`` is split across layers. Each layer keeps its own hot set.
    ``overlap=True`` prefetches on a background thread.
    """

    def __init__(
        self,
        directory: str | Path,
        budget_bytes: int,
        layer_ids: list[int],
        *,
        overlap: bool = True,
    ) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be >= 0, got {budget_bytes}")
        if not layer_ids:
            raise ValueError("layer_ids must name at least one layer")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"layer_ids must be unique, got {layer_ids}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layer_ids = list(layer_ids)
        self.n_layers = len(self.layer_ids)
        base, extra = divmod(int(budget_bytes), self.n_layers)
        self._budgets = {
            layer: base + (1 if index < extra else 0) for index, layer in enumerate(self.layer_ids)
        }
        self.capacity_bytes = sum(self._budgets.values())
        self._sizes: dict[tuple[int, int], int] = {}
        self._slots: dict[int, list[_Slot]] = {layer: [] for layer in self.layer_ids}
        self._heat: dict[int, dict[int, int]] = {layer: {} for layer in self.layer_ids}
        self._last: dict[int, dict[int, int]] = {layer: {} for layer in self.layer_ids}
        self._clock: dict[int, int] = {layer: 0 for layer in self.layer_ids}
        self._previous: dict[int, list[int]] = {}
        self._stats = ExpertStoreStats(capacity_bytes=self.capacity_bytes)
        self._lock = threading.RLock()
        self._overlap = overlap
        self._queue: queue.Queue[list[tuple[int, int]] | None] | None = None
        self._thread: threading.Thread | None = None
        if overlap:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._worker, name="yami-expert-prefetch", daemon=True)
            self._thread.start()

    def spill(self, layer: int, expert: int, state: dict[str, torch.Tensor]) -> None:
        """Write one expert to disk. The tensors are not kept in RAM."""
        self._check_layer(layer)
        payload = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
        path = self._path(layer, expert)
        torch.save(payload, path)
        with self._lock:
            self._sizes[(layer, expert)] = _nbytes(payload)
            self._heat[layer].setdefault(expert, 0)
            self._last[layer].setdefault(expert, 0)
        del payload

    def lookup(self, layer: int, expert: int) -> _View:
        """Return weights for ``(layer, expert)`` and hold one lease."""
        self._check_layer(layer)
        with self._lock:
            self._stats.requests += 1
            slot = self._find(layer, expert)
            if slot is not None:
                self._stats.hits += 1
                self._bump(layer, expert)
                slot.leases += 1
                return _View(layer, expert, slot.weights, slot)
        weights = self._read(layer, expert)
        nbytes = _nbytes(weights)
        with self._lock:
            slot = self._find(layer, expert)
            if slot is not None:
                self._stats.hits += 1
                self._bump(layer, expert)
                slot.leases += 1
                return _View(layer, expert, slot.weights, slot)
            self._stats.misses += 1
            self._stats.bytes_read += nbytes
            self._bump(layer, expert)
            if self._try_admit(layer, expert, weights, nbytes):
                slot = self._find(layer, expert)
                assert slot is not None
                slot.leases = 1
                return _View(layer, expert, slot.weights, slot)
            return _View(layer, expert, weights, None)

    def release(self, view: _View) -> None:
        """Drop one lease. A second release of the same view does nothing."""
        with self._lock:
            if view.released:
                return
            view.released = True
            if view.slot is None:
                view.weights.clear()
                return
            view.slot.leases -= 1
            if view.slot.leases < 0:
                raise RuntimeError(
                    f"expert layer={view.layer} expert={view.expert} released more times than it was leased"
                )

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        """Advise the cache to load ``keys``. Holds no lease."""
        if not keys or self.capacity_bytes <= 0:
            return
        if self._queue is not None:
            self._queue.put(list(keys))
            return
        self._prefetch_now(keys)

    def prefetch_previous(self, layer: int) -> None:
        """Load the experts this layer routed on the previous step."""
        chosen = self._previous.get(layer)
        if not chosen:
            return
        self.prefetch([(layer, expert) for expert in chosen])

    def remember(self, layer: int, experts: list[int]) -> None:
        """Record the experts routed at ``layer`` for the next step's prefetch."""
        self._previous[layer] = list(experts)

    def run(
        self,
        layer: int,
        expert: int,
        inputs: torch.Tensor,
        activation: str,
        kernel: str,
    ) -> torch.Tensor:
        """Lookup, run, and release one expert."""
        if inputs.requires_grad:
            raise RuntimeError(
                "expert streaming is an inference path; routed experts are not in the autograd graph"
            )
        view = self.lookup(layer, expert)
        try:
            return feedforward_from_state(view.weights, inputs, activation, kernel)
        finally:
            self.release(view)

    def stats(self) -> ExpertStoreStats:
        """Return a snapshot of the counters."""
        with self._lock:
            resident = sum(slot.nbytes for slots in self._slots.values() for slot in slots)
            return ExpertStoreStats(
                requests=self._stats.requests,
                hits=self._stats.hits,
                misses=self._stats.misses,
                prefetched=self._stats.prefetched,
                prefetch_hits=self._stats.prefetch_hits,
                bytes_read=self._stats.bytes_read,
                resident_bytes=resident,
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

    def _worker(self) -> None:
        assert self._queue is not None
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            try:
                self._prefetch_now(batch)
            except Exception:
                logger.exception("expert prefetch failed")

    def _prefetch_now(self, keys: list[tuple[int, int]]) -> None:
        for layer, expert in keys:
            if layer not in self._budgets:
                continue
            with self._lock:
                if self._find(layer, expert) is not None:
                    self._stats.prefetched += 1
                    self._stats.prefetch_hits += 1
                    continue
            try:
                weights = self._read(layer, expert)
            except FileNotFoundError:
                continue
            nbytes = _nbytes(weights)
            with self._lock:
                if self._find(layer, expert) is not None:
                    self._stats.prefetch_hits += 1
                else:
                    self._stats.bytes_read += nbytes
                    self._heat[layer].setdefault(expert, 0)
                    self._last[layer].setdefault(expert, 0)
                    self._try_admit(layer, expert, weights, nbytes)
                self._stats.prefetched += 1

    def _try_admit(self, layer: int, expert: int, weights: dict[str, torch.Tensor], nbytes: int) -> bool:
        budget = self._budgets[layer]
        if nbytes > budget or nbytes <= 0:
            return False
        slots = self._slots[layer]
        while self._used(layer) + nbytes > budget:
            victim = self._coldest_unleased(layer)
            if victim is None or not self._beats(layer, expert, victim.expert):
                return False
            slots.remove(victim)
        slots.append(_Slot(layer, expert, weights, nbytes))
        return True

    def _beats(self, layer: int, candidate: int, victim: int) -> bool:
        clock = self._clock[layer]
        hot = tier_lfru_score(self._heat[layer][candidate], self._last[layer][candidate], clock)
        cold = tier_lfru_score(self._heat[layer][victim], self._last[layer][victim], clock)
        return hot > cold + (cold >> 2) + (4 << 8)

    def _coldest_unleased(self, layer: int) -> _Slot | None:
        clock = self._clock[layer]
        victim: _Slot | None = None
        victim_score = 0
        for slot in self._slots[layer]:
            if slot.leases:
                continue
            score = tier_lfru_score(self._heat[layer][slot.expert], self._last[layer][slot.expert], clock)
            if victim is None or score < victim_score:
                victim = slot
                victim_score = score
        return victim

    def _used(self, layer: int) -> int:
        return sum(slot.nbytes for slot in self._slots[layer])

    def _bump(self, layer: int, expert: int) -> None:
        clock = (self._clock[layer] + 1) & _U32
        self._clock[layer] = clock
        if clock % _DECAY_EVERY == 0:
            heat = self._heat[layer]
            for expert_id in list(heat):
                heat[expert_id] = tier_decay_value(heat[expert_id])
        heat_value = self._heat[layer].get(expert, 0)
        if heat_value < _U32:
            heat_value += 1
        self._heat[layer][expert] = heat_value
        self._last[layer][expert] = clock

    def _find(self, layer: int, expert: int) -> _Slot | None:
        for slot in self._slots[layer]:
            if slot.expert == expert:
                return slot
        return None

    def _read(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        path = self._path(layer, expert)
        if not path.is_file():
            raise FileNotFoundError(f"missing streamed expert layer={layer} expert={expert} at {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"streamed expert {path} is not a state dict")
        return payload

    def _path(self, layer: int, expert: int) -> Path:
        return self.directory / f"layer{layer:04d}_expert{expert:05d}.pt"

    def _check_layer(self, layer: int) -> None:
        if layer not in self._budgets:
            raise KeyError(f"layer {layer} is not in this expert store")


class StreamedExpert(nn.Module):
    """Placeholder left in the module tree after weights move to the store."""

    def __init__(self, expert_id: int) -> None:
        super().__init__()
        self.expert_id = expert_id

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - store runs the expert
        raise RuntimeError(
            f"expert {self.expert_id} was spilled to disk; run it through the expert store"
        )


def attach_expert_streaming(
    model: nn.Module,
    budget_bytes: int,
    directory: str | Path,
    *,
    overlap: bool = True,
) -> ExpertStore | None:
    """Spill routed experts to ``directory`` and serve them from a RAM budget.

    Embeddings, attention, norms, routers, and shared experts stay parameters.
    Returns ``None`` when the model has no mixture-of-experts layers. Training
    must not call this: the spilled experts leave the autograd graph.
    """
    blocks: list[tuple[int, MixtureOfExperts]] = []
    for module in model.modules():
        if isinstance(module, MixtureOfExperts) and module.expert_store is None:
            blocks.append((int(module.layer_index), module))
    if not blocks:
        return None
    layer_ids = [layer for layer, _ in blocks]
    store = ExpertStore(directory, budget_bytes, layer_ids, overlap=overlap)
    for layer, moe in blocks:
        for local_index, expert_id in enumerate(moe.global_expert_ids):
            state = moe.experts[local_index].state_dict()
            store.spill(layer, int(expert_id), state)
        moe.experts = nn.ModuleList(StreamedExpert(int(expert_id)) for expert_id in moe.global_expert_ids)
        moe.expert_store = store
        moe.layer_index = layer
    logger.info(
        "expert streaming layers=%d budget_bytes=%d directory=%s",
        store.n_layers,
        store.capacity_bytes,
        store.directory,
    )
    return store
