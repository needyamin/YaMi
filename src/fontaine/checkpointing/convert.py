"""Checkpoint conversion utilities.

The native format (``.pt`` files + ``meta.json``) is the canonical one. Export
targets keep Fontaine interoperable:

- ``safetensors``: safe (no pickle) tensor serialization, the standard for
  publishing weights. Requires the optional ``safetensors`` package.

Future targets (Hugging Face format, sharded/distributed layouts) register
here; ``fontaine checkpoint convert`` stays the single entry point.
"""

from pathlib import Path

import torch

from fontaine.checkpointing.errors import CheckpointError
from fontaine.checkpointing.manager import CheckpointManager
from fontaine.utils.io import ensure_dir


def convert_checkpoint(
    checkpoint: str | Path, target_format: str = "safetensors", output: str | Path | None = None
) -> Path:
    """Export the model weights of a checkpoint to another format."""
    checkpoint = Path(checkpoint)
    root = checkpoint.parent if (checkpoint / "meta.json").is_file() else checkpoint
    manager = CheckpointManager(root)
    source = checkpoint if (checkpoint / "meta.json").is_file() else "latest"

    if target_format == "safetensors":
        return _convert_to_safetensors(manager, source, output)
    raise CheckpointError(f"unsupported conversion target: {target_format!r}")


def _convert_to_safetensors(manager: CheckpointManager, source, output: str | Path | None) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CheckpointError(
            "safetensors export requires the optional dependency: "
            "pip install safetensors (or pip install fontaine-ai[safetensors])"
        ) from exc

    manager.load(source)  # integrity check happens inside load()
    step_dir = manager._resolve(source)
    output_path = Path(output) if output else step_dir / "model.safetensors"

    payload = torch.load(step_dir / "model.pt", map_location="cpu")
    state_dict = payload["model"]
    # safetensors requires contiguous tensors and no shared storage; the tied
    # lm_head/embedding pair is materialized explicitly.
    clean: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        clean[key] = tensor.detach().clone().contiguous()
    ensure_dir(output_path.parent)
    save_file(clean, str(output_path))
    return output_path
