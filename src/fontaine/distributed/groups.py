"""Process groups for a combined data/tensor/pipeline/expert mesh.

Every rank must call :func:`init_parallel_groups` so the groups are created
in the same order. An axis of size 1 does not get a group; the matching
collective is skipped.
"""

from typing import Any

import torch
import torch.distributed as dist

from fontaine.distributed.topology import ParallelTopology

_groups: dict[str, Any] = {"tensor": None, "pipeline": None, "expert": None, "data": None}


def tensor_group() -> Any:
    return _groups["tensor"]


def expert_group() -> Any:
    return _groups["expert"]


def data_group() -> Any:
    return _groups["data"]


def init_parallel_groups(topology: ParallelTopology) -> ParallelTopology:
    """Create groups and return the topology annotated for this rank."""
    if not (dist.is_available() and dist.is_initialized()):
        if topology.world_size != 1:
            raise RuntimeError(
                f"distributed world_size is {topology.world_size} but torch.distributed "
                "is not initialized. Launch with torchrun. Training was not started."
            )
        _clear()
        return topology.for_rank(0)
    local = topology.for_rank(dist.get_rank())
    _clear()
    _groups["tensor"] = _tensor_group(topology, local)
    _groups["pipeline"] = _pipeline_group(topology, local)
    _groups["expert"] = _expert_group(topology, local)
    _groups["data"] = _data_group(topology, local)
    return local


def sync_data_parallel_gradients(model: torch.nn.Module) -> None:
    """Average gradients across the data-parallel group."""
    group = _groups["data"]
    if group is None or not (dist.is_available() and dist.is_initialized()):
        return
    size = dist.get_world_size(group)
    if size < 2:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, group=group)
        param.grad.div_(size)


def _clear() -> None:
    for key in _groups:
        _groups[key] = None


def _tensor_group(topology: ParallelTopology, local: ParallelTopology):
    if topology.tensor_parallel < 2:
        return None
    found = None
    for data_rank in range(topology.data_parallel):
        for pipeline_rank in range(topology.pipeline_parallel):
            ranks = [
                topology.global_rank(data_rank, pipeline_rank, tensor_rank)
                for tensor_rank in range(topology.tensor_parallel)
            ]
            group = dist.new_group(ranks)
            if data_rank == local.data_rank and pipeline_rank == local.pipeline_rank:
                found = group
    return found


def _pipeline_group(topology: ParallelTopology, local: ParallelTopology):
    if topology.pipeline_parallel < 2:
        return None
    found = None
    for data_rank in range(topology.data_parallel):
        for tensor_rank in range(topology.tensor_parallel):
            ranks = [
                topology.global_rank(data_rank, pipeline_rank, tensor_rank)
                for pipeline_rank in range(topology.pipeline_parallel)
            ]
            group = dist.new_group(ranks)
            if data_rank == local.data_rank and tensor_rank == local.tensor_rank:
                found = group
    return found


def _expert_group(topology: ParallelTopology, local: ParallelTopology):
    if topology.expert_parallel < 2:
        return None
    found = None
    buckets = topology.data_parallel // topology.expert_parallel
    for bucket in range(buckets):
        for pipeline_rank in range(topology.pipeline_parallel):
            for tensor_rank in range(topology.tensor_parallel):
                ranks = [
                    topology.global_rank(
                        bucket * topology.expert_parallel + expert_rank,
                        pipeline_rank,
                        tensor_rank,
                    )
                    for expert_rank in range(topology.expert_parallel)
                ]
                group = dist.new_group(ranks)
                if (
                    local.data_rank // topology.expert_parallel == bucket
                    and pipeline_rank == local.pipeline_rank
                    and tensor_rank == local.tensor_rank
                ):
                    found = group
    return found


def _data_group(topology: ParallelTopology, local: ParallelTopology):
    """Ranks that hold the same shard and differ only by the data-parallel replica."""
    buckets = topology.data_parallel // topology.expert_parallel
    if buckets < 2:
        return None
    found = None
    for expert_rank in range(topology.expert_parallel):
        for pipeline_rank in range(topology.pipeline_parallel):
            for tensor_rank in range(topology.tensor_parallel):
                ranks = [
                    topology.global_rank(
                        bucket * topology.expert_parallel + expert_rank,
                        pipeline_rank,
                        tensor_rank,
                    )
                    for bucket in range(buckets)
                ]
                group = dist.new_group(ranks)
                if (
                    local.expert_rank == expert_rank
                    and pipeline_rank == local.pipeline_rank
                    and tensor_rank == local.tensor_rank
                ):
                    found = group
    return found
