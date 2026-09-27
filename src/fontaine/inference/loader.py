"""Loading a Generator from a checkpoint + tokenizer directory."""

from pathlib import Path

import torch

from fontaine.checkpointing import CheckpointManager
from fontaine.config.loader import dataclass_from_dict
from fontaine.config.schema import InferenceConfig, ModelConfig
from fontaine.inference.engine import Generator
from fontaine.models import FontaineModel
from fontaine.tokenizer.base import Tokenizer
from fontaine.tokenizer.registry import load_tokenizer


def load_generator(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    inference_config: InferenceConfig | None = None,
    device: str = "auto",
    model_overrides: dict | None = None,
    distributed: object | None = None,
) -> Generator:
    """Build a Generator from a checkpoint directory and a trained tokenizer.

    The model architecture is restored from the architecture snapshot stored
    inside the checkpoint (``meta.json``), so inference never depends on the
    user remembering which config built the model. ``model_overrides`` allows
    surgical fixes for experimental checkpoints.
    """
    checkpoint_path = Path(checkpoint_path)
    if (checkpoint_path / "meta.json").is_file():
        # A specific checkpoint step directory: e.g. .../checkpoints/step_000001000
        root, source = checkpoint_path.parent, checkpoint_path
    elif (checkpoint_path / "latest.json").is_file():
        # A CheckpointManager root directory: resolve via the 'latest' pointer.
        root, source = checkpoint_path, "latest"
    else:
        raise FileNotFoundError(
            f"{checkpoint_path} is neither a checkpoint step directory (meta.json) "
            f"nor a checkpoints root (latest.json)"
        )
    manager = CheckpointManager(root)
    meta = manager.inspect(source)
    model_config_dict = dict(meta.get("config", {}).get("model", {}))
    if not model_config_dict:
        raise ValueError(
            f"checkpoint {checkpoint_path} has no stored model configuration; "
            f"pass model_overrides explicitly"
        )
    model_config_dict.update(model_overrides or {})
    model_config = dataclass_from_dict(ModelConfig, model_config_dict)

    model = _load_model(manager, source, model_config, meta, distributed)
    if device != "cpu" and not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("requested device 'cuda' is not available")

    tokenizer: Tokenizer = load_tokenizer(tokenizer_path)
    draft = _load_draft(inference_config, tokenizer_path, device)
    return Generator(model, tokenizer, inference_config, device=device, draft=draft)


def _load_draft(
    inference_config: InferenceConfig | None,
    tokenizer_path: str | Path,
    device: str,
) -> FontaineModel | None:
    if inference_config is None or not inference_config.draft_checkpoint:
        return None
    quiet = InferenceConfig(
        precision=inference_config.precision,
        num_threads=inference_config.num_threads,
        kv_cache=inference_config.kv_cache,
        max_new_tokens=inference_config.max_new_tokens,
    )
    return load_generator(
        inference_config.draft_checkpoint,
        tokenizer_path,
        quiet,
        device=device,
    ).model


def _load_model(manager, source, model_config: ModelConfig, meta: dict, distributed: object | None):
    """Load weights, slicing them when tensor parallel is requested."""
    world = int(getattr(distributed, "world_size", 1) or 1)
    if distributed is None or world == 1:
        model = FontaineModel(model_config)
        manager.load(source, model=model, map_location="cpu")
        return model
    from fontaine.distributed.groups import init_parallel_groups
    from fontaine.distributed.topology import topology_from_config, validate_runtime_topology

    validate_runtime_topology(distributed)  # type: ignore[arg-type]
    local = init_parallel_groups(topology_from_config(distributed))  # type: ignore[arg-type]
    checkpoint_world = int(meta.get("world_size", 1))
    if checkpoint_world == 1 and (local.pipeline_parallel > 1 or local.expert_parallel > 1):
        raise RuntimeError(
            "this checkpoint stores the full model (world_size=1). "
            f"pipeline_parallel_size={local.pipeline_parallel} and "
            f"expert_parallel_size={local.expert_parallel} need a checkpoint saved with "
            "that mesh. Tensor parallel can slice a full checkpoint after it is loaded."
        )
    use_mesh = local.pipeline_parallel > 1 or local.expert_parallel > 1 or checkpoint_world > 1
    model = FontaineModel(model_config, mesh=local if use_mesh else None)
    if local.tensor_parallel > 1 and checkpoint_world > 1:
        _shard_linears(model, local)
    manager.load(source, model=model, map_location="cpu")
    if local.tensor_parallel > 1 and checkpoint_world == 1:
        _shard_linears(model, local)
    return model


def _shard_linears(model: FontaineModel, local) -> None:
    import torch.distributed as dist

    from fontaine.distributed.groups import tensor_group
    from fontaine.distributed.tensor_parallel import TensorParallelLinear, apply_tensor_parallel

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "tensor_parallel_size > 1 needs an initialized process group. Launch with torchrun."
        )
    apply_tensor_parallel(model, local.tensor_rank, local.tensor_parallel)
    group = tensor_group()
    for module in model.modules():
        if isinstance(module, TensorParallelLinear):
            module.process_group = group
