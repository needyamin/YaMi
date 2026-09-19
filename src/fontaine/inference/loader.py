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

    model = FontaineModel(model_config)
    manager.load(source, model=model, map_location="cpu")
    if device != "cpu" and not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("requested device 'cuda' is not available")

    tokenizer: Tokenizer = load_tokenizer(tokenizer_path)
    return Generator(model, tokenizer, inference_config, device=device)
