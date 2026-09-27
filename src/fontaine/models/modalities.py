"""Modalities that actually run.

Text is the embedding table. Image, audio, and video are not implemented;
requesting them fails in config validation and in ``require_modality``.
"""

import torch

from fontaine.models.transformer import FontaineModel


def require_modality(name: str) -> None:
    if name != "text":
        raise ValueError(
            f"modality {name!r} is not implemented. YaMi currently runs text only."
        )


def embed_text(model: FontaineModel, input_ids: torch.Tensor) -> torch.Tensor:
    require_modality("text")
    if model.token_embedding is None:
        raise RuntimeError("this pipeline stage does not own the token embedding")
    return model.token_embedding(input_ids)
