"""Decide whether a configuration fits the machine without allocating the model."""

import math
from dataclasses import dataclass

from fontaine.config.schema import FontaineConfig
from fontaine.distributed.topology import ParallelTopology, plan_topology
from fontaine.optimization.hardware import HardwareInfo
from fontaine.optimization.memory import (
    estimate_inference_memory,
    estimate_kv_cache_bytes,
    estimate_parameter_count,
    estimate_training_memory,
)


@dataclass
class FeasibilityReport:
    inference: str
    training: str
    fine_tuning: str
    gpu_memory_bytes: int
    system_memory_bytes: int
    accelerators: int
    topology: ParallelTopology
    checkpoint_bytes: int
    notes: list[str]

    def lines(self) -> list[str]:
        from fontaine.optimization.memory import format_bytes

        topo = self.topology
        return [
            f"inference: {self.inference}",
            f"training: {self.training}",
            f"fine-tuning: {self.fine_tuning}",
            f"GPU memory: {format_bytes(self.gpu_memory_bytes)}",
            f"system memory: {format_bytes(self.system_memory_bytes)}",
            f"accelerators: {self.accelerators}",
            (
                f"parallelism: data={topo.data_parallel} tensor={topo.tensor_parallel} "
                f"pipeline={topo.pipeline_parallel} expert={topo.expert_parallel}"
            ),
            f"checkpoint size: {format_bytes(self.checkpoint_bytes)}",
            *self.notes,
        ]


def assess(config: FontaineConfig, hardware: HardwareInfo, world_size: int | None = None) -> FeasibilityReport:
    counts = estimate_parameter_count(config.model)
    sequence = min(config.data.sequence_length, config.model.max_sequence_length)
    training = estimate_training_memory(
        config.model,
        batch_size=config.training.batch_size,
        sequence_length=sequence,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    inference = estimate_inference_memory(config.model, "bf16" if hardware.bf16 else "fp32")
    devices = world_size or hardware.gpu_count or 1
    topology = plan_topology(config.model, devices)
    per_device = _per_device_bytes(training.total_bytes, counts, topology, config.model.num_experts)
    gpu_budget = hardware.gpus[0].memory_bytes if hardware.gpus else 0
    system_budget = hardware.available_ram_bytes or hardware.total_ram_bytes or 0
    notes: list[str] = []
    if gpu_budget:
        inference_ok = inference["total"] <= gpu_budget * topology.tensor_parallel * topology.pipeline_parallel
        train_ok = per_device <= gpu_budget
    else:
        inference_ok = inference["total"] <= system_budget if system_budget else False
        train_ok = training.total_bytes <= system_budget if system_budget else False
        notes.append("no GPU was detected; estimates use system memory")
    if not system_budget and not gpu_budget:
        notes.append("memory capacity could not be read; feasibility is unknown")
        inference_text = "unknown"
        train_text = "unknown"
    else:
        inference_text = "feasible" if inference_ok else "not feasible"
        train_text = "feasible" if train_ok else "not feasible"
    notes.append(
        "fine-tuning uses the same full-weight AdamW estimate as training. "
        "Parameter-efficient fine-tuning is not implemented."
    )
    notes.append(
        f"KV cache at the full context is { _fmt(estimate_kv_cache_bytes(config.model)) } "
        "and is included in the inference estimate, not in the training total."
    )
    checkpoint = 4 * counts["total"]
    return FeasibilityReport(
        inference=inference_text,
        training=train_text,
        fine_tuning=train_text,
        gpu_memory_bytes=per_device if gpu_budget else 0,
        system_memory_bytes=training.total_bytes,
        accelerators=devices,
        topology=topology,
        checkpoint_bytes=checkpoint,
        notes=notes,
    )


def _per_device_bytes(training_bytes: int, counts: dict[str, int], topology: ParallelTopology, num_experts: int) -> int:
    split = max(1, topology.tensor_parallel * topology.pipeline_parallel)
    if num_experts > 1 and counts["total"]:
        expert_share = counts["moe_experts"] / counts["total"]
        other = 1.0 - expert_share
        expert_split = split * max(1, topology.expert_parallel)
        fraction = other / split + expert_share / expert_split
        return max(1, math.ceil(training_bytes * fraction))
    return max(1, math.ceil(training_bytes / split))


def _fmt(num_bytes: int) -> str:
    from fontaine.optimization.memory import format_bytes

    return format_bytes(num_bytes)
