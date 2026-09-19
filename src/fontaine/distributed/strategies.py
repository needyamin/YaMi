"""Distributed-training seams for future scale-up.

The first Fontaine release is single-process: the trainer accepts a
``TrainingStrategy`` and a ``ParallelContext`` that are trivial no-ops today.
The interfaces are the migration path — when Fontaine grows to multi-GPU
(DDP), multi-node FSDP, or tensor/pipeline parallelism, the *trainer loop*
stays untouched; only the strategy implementation changes.

Framework notes (see ``docs/scaling/roadmap.md``):
- DDP: torch.distributed + DistributedSampler (skeleton provided).
- FSDP / DeepSpeed ZeRO: shard parameters/optimizer/activations across ranks.
- Megatron-style TP/PP: requires model sharding inside the Transformer —
  the config-driven model is structured for it but not implemented.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from fontaine.utils.logging import get_logger

logger = get_logger("distributed")


@dataclass
class ParallelContext:
    """Where this process sits in the (future) distributed job."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"

    @property
    def is_primary(self) -> bool:
        """Only the primary rank writes checkpoints, logs, and manifests."""
        return self.rank == 0

    @classmethod
    def single_process(cls) -> "ParallelContext":
        return cls()

    @classmethod
    def init_from_env(cls) -> "ParallelContext":
        """Join an existing torch.distributed job (torchrun-style env vars)."""
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ParallelContext.init_from_env requires an initialized process group "
                "(launch with torchrun)"
            )
        return cls(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=int(torch.cuda.current_device()) if torch.cuda.is_available() else 0,
            backend=dist.get_backend(),
        )

    def destroy(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class TrainingStrategy:
    """Base strategy: single-process behavior (the Phase 1 default).

    Subclasses override the hooks to add parallelism; the trainer calls only
    these hooks, never torch.distributed directly.
    """

    name = "single_device"

    def __init__(self, context: ParallelContext | None = None) -> None:
        self.context = context or ParallelContext.single_process()

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        return optimizer

    def sync_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Reduce metrics across ranks (mean); no-op single-process."""
        return metrics

    def wait_for_everyone(self) -> None:
        return None


class DistributedDataParallelStrategy(TrainingStrategy):
    """Standard DDP wrapper (Phase 8 — multi-GPU on one node).

    Requires launching via ``torchrun``; gradients synchronize every step.
    For models too large for full replication, FSDP/DeepSpeed strategies are
    the next step — they slot in behind the same interface.
    """

    name = "ddp"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "DistributedDataParallelStrategy requires torch.distributed; "
                "launch with torchrun and call dist.init_process_group first"
            )
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    def sync_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not (dist.is_available() and dist.is_initialized()):
            return metrics
        synced: dict[str, float] = {}
        for key, value in metrics.items():
            tensor = torch.tensor(float(value), dtype=torch.float64)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced[key] = tensor.item() / dist.get_world_size()
        return synced

    def wait_for_everyone(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def unwrap_model(model: Any) -> torch.nn.Module:
    """Return the underlying module (DistributedDataParallel strips .module)."""
    return model.module if hasattr(model, "module") else model
