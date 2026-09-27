"""Parallel topology validation and a planner that does not launch training.

World size is ``data_parallel * tensor_parallel * pipeline_parallel``.
Expert parallel divides the data-parallel group. Sequence parallel is 1 or
equal to tensor parallel.
"""

from dataclasses import dataclass

import torch.distributed as dist

from fontaine.config.errors import ConfigError
from fontaine.config.schema import DistributedConfig, ModelConfig


@dataclass(frozen=True)
class ParallelTopology:
    data_parallel: int
    tensor_parallel: int
    pipeline_parallel: int
    sequence_parallel: int
    expert_parallel: int
    # Ranks are filled by ``for_rank``. A planned mesh leaves them at 0.
    data_rank: int = 0
    pipeline_rank: int = 0
    tensor_rank: int = 0
    expert_rank: int = 0

    @property
    def world_size(self) -> int:
        return self.data_parallel * self.tensor_parallel * self.pipeline_parallel

    def global_rank(self, data_rank: int, pipeline_rank: int, tensor_rank: int) -> int:
        """Rank layout: data, then pipeline, then tensor (tensor ranks are adjacent)."""
        return (
            (data_rank * self.pipeline_parallel + pipeline_rank) * self.tensor_parallel
        ) + tensor_rank

    def for_rank(self, rank: int) -> "ParallelTopology":
        """Return this mesh with the coordinates of one process."""
        if rank < 0 or rank >= self.world_size:
            raise ConfigError(
                f"rank {rank} is outside 0..{self.world_size - 1} for this topology"
            )
        tensor_rank = rank % self.tensor_parallel
        rest = rank // self.tensor_parallel
        pipeline_rank = rest % self.pipeline_parallel
        data_rank = rest // self.pipeline_parallel
        return ParallelTopology(
            data_parallel=self.data_parallel,
            tensor_parallel=self.tensor_parallel,
            pipeline_parallel=self.pipeline_parallel,
            sequence_parallel=self.sequence_parallel,
            expert_parallel=self.expert_parallel,
            data_rank=data_rank,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
            expert_rank=data_rank % self.expert_parallel,
        )

    def pipeline_prev(self) -> int | None:
        if self.pipeline_rank == 0:
            return None
        return self.global_rank(self.data_rank, self.pipeline_rank - 1, self.tensor_rank)

    def pipeline_next(self) -> int | None:
        if self.pipeline_rank + 1 >= self.pipeline_parallel:
            return None
        return self.global_rank(self.data_rank, self.pipeline_rank + 1, self.tensor_rank)

    def as_dict(self) -> dict[str, int]:
        return {
            "data_parallel": self.data_parallel,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "sequence_parallel": self.sequence_parallel,
            "expert_parallel": self.expert_parallel,
            "world_size": self.world_size,
        }

    def to_config(self) -> DistributedConfig:
        return DistributedConfig(
            data_parallel_size=self.data_parallel,
            tensor_parallel_size=self.tensor_parallel,
            pipeline_parallel_size=self.pipeline_parallel,
            sequence_parallel_size=self.sequence_parallel,
            expert_parallel_size=self.expert_parallel,
        )


def topology_from_config(config: DistributedConfig) -> ParallelTopology:
    return ParallelTopology(
        data_parallel=config.data_parallel_size,
        tensor_parallel=config.tensor_parallel_size,
        pipeline_parallel=config.pipeline_parallel_size,
        sequence_parallel=config.sequence_parallel_size,
        expert_parallel=config.expert_parallel_size,
    )


def runtime_world_size() -> int:
    """Process count of the current job. A process that did not join a group has size 1."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def validate_runtime_topology(config: DistributedConfig) -> None:
    """Fail when the configured mesh does not match this process group."""
    expected = config.world_size
    actual = runtime_world_size()
    if expected == actual:
        return
    if actual == 1 and expected > 1:
        raise RuntimeError(
            f"distributed world_size is {expected} "
            f"(data={config.data_parallel_size} x tensor={config.tensor_parallel_size} "
            f"x pipeline={config.pipeline_parallel_size}) but this process is not in a "
            f"distributed job. Launch with torchrun --nproc_per_node={expected} "
            f"so the topology matches the process group. "
            f"Training was not started."
        )
    raise RuntimeError(
        f"distributed world_size is {expected} but the process group has {actual} ranks. "
        f"Set data_parallel_size * tensor_parallel_size * pipeline_parallel_size "
        f"equal to the launcher's world size."
    )


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def plan_topology(config: ModelConfig, world_size: int) -> ParallelTopology:
    """Choose a mesh that divides ``world_size`` and the model dimensions.

    The search prefers a split that reduces parameters held on one device
    (tensor, pipeline, and expert parallel) and leaves the remainder as data
    parallel. It does not start training.
    """
    if world_size < 1:
        raise ConfigError(f"world_size must be >= 1, got {world_size}")
    if world_size == 1:
        return ParallelTopology(1, 1, 1, 1, 1)

    candidates: list[ParallelTopology] = []
    max_pp = min(world_size, config.num_layers)
    for pp in range(1, max_pp + 1):
        if world_size % pp != 0:
            continue
        rest = world_size // pp
        for tp in _divisors(rest):
            if config.num_attention_heads % tp != 0 or config.num_kv_heads % tp != 0:
                continue
            if config.intermediate_size % tp != 0:
                continue
            expert_width = config.resolved_expert_intermediate()
            if config.num_experts > 1 and expert_width % tp != 0:
                continue
            dp = rest // tp
            if config.num_experts > 1:
                ep_choices = [ep for ep in _divisors(dp) if config.num_experts % ep == 0]
            else:
                ep_choices = [1]
            for ep in ep_choices:
                sp = tp if tp > 1 and config.max_sequence_length % tp == 0 else 1
                candidates.append(ParallelTopology(dp, tp, pp, sp, ep))
    if not candidates:
        raise ConfigError(
            f"no parallelism topology divides world_size={world_size} for "
            f"num_attention_heads={config.num_attention_heads}, "
            f"num_kv_heads={config.num_kv_heads}, num_layers={config.num_layers}, "
            f"intermediate_size={config.intermediate_size}, "
            f"num_experts={config.num_experts}. "
            f"Choose a world size whose factors divide those dimensions."
        )

    def memory_key(topo: ParallelTopology) -> tuple[float, int]:
        expert_split = topo.expert_parallel if config.num_experts > 1 else 1
        per_device = 1.0 / (topo.tensor_parallel * topo.pipeline_parallel * expert_split)
        return (per_device, -topo.data_parallel)

    return min(candidates, key=memory_key)


def layer_stage(num_layers: int, pipeline_parallel: int, stage: int) -> range:
    """Layer indices owned by one pipeline stage. Earlier stages take the remainder."""
    if stage < 0 or stage >= pipeline_parallel:
        raise ConfigError(
            f"pipeline stage {stage} is outside 0..{pipeline_parallel - 1}"
        )
    base, extra = divmod(num_layers, pipeline_parallel)
    start = 0
    for index in range(pipeline_parallel):
        count = base + (1 if index < extra else 0)
        if index == stage:
            return range(start, start + count)
        start += count
    raise ConfigError(f"pipeline stage {stage} was not assigned any layers")


def expert_ids_for_rank(num_experts: int, expert_parallel: int, rank: int) -> range:
    """Contiguous global expert ids owned by one expert-parallel rank."""
    if num_experts % expert_parallel != 0:
        raise ConfigError(
            f"expert_parallel={expert_parallel} must divide num_experts={num_experts}"
        )
    if rank < 0 or rank >= expert_parallel:
        raise ConfigError(f"expert rank {rank} is outside 0..{expert_parallel - 1}")
    per = num_experts // expert_parallel
    start = rank * per
    return range(start, start + per)
