"""Model construction factory.

``build_model`` is the only place that maps an ``architecture`` selector to a
concrete class. Future architectures (mixture-of-experts, state-space models)
register here, keeping every other subsystem (trainer, checkpointing,
inference) architecture-agnostic.
"""

from fontaine.config.schema import ModelConfig
from fontaine.models.transformer import FontaineModel


def build_model(config: ModelConfig) -> FontaineModel:
    """Build a model instance from validated configuration."""
    if config.architecture == "decoder_transformer":
        return FontaineModel(config)
    raise ValueError(
        f"unknown architecture {config.architecture!r} — register it in "
        f"fontaine.models.factory to extend Fontaine"
    )
