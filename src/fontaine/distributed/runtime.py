"""Build a model whose parallelism matches ``FontaineConfig.distributed``."""

import torch.distributed as dist

from fontaine.config.schema import FontaineConfig
from fontaine.distributed.groups import expert_group, init_parallel_groups, tensor_group
from fontaine.distributed.tensor_parallel import TensorParallelLinear, apply_tensor_parallel
from fontaine.distributed.topology import topology_from_config, validate_runtime_topology
from fontaine.models.components import MixtureOfExperts
from fontaine.models.factory import build_model
from fontaine.models.transformer import FontaineModel


def build_trainable_model(config: FontaineConfig) -> FontaineModel:
    """Construct the model for this rank.

    Tensor-parallel linears are sliced, pipeline stages keep only their layers,
    and expert parallel keeps only the local experts. A world size of 1 builds
    the same module as ``build_model``.
    """
    validate_runtime_topology(config.distributed)
    local = init_parallel_groups(topology_from_config(config.distributed))
    model = build_model(config.model, mesh=local)
    if local.tensor_parallel > 1:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != local.global_rank(local.data_rank, local.pipeline_rank, local.tensor_rank):
            raise RuntimeError(
                "tensor-parallel rank does not match the process rank. "
                "The mesh layout is data, then pipeline, then tensor."
            )
        apply_tensor_parallel(model, local.tensor_rank, local.tensor_parallel)
        group = tensor_group()
        for module in model.modules():
            if isinstance(module, TensorParallelLinear):
                module.process_group = group
    group = expert_group()
    for module in model.modules():
        if isinstance(module, MixtureOfExperts):
            module.parallel_group = group
    return model
