"""Model construction factory.

``build_model`` is the only place that maps an ``architecture`` selector to a
concrete class. ``decoder_transformer`` and ``moe_decoder`` share
``FontaineModel``; the mixture-of-experts path is selected by config
(``num_experts``). Further architectures register here, keeping every other
subsystem (trainer, checkpointing, inference) architecture-agnostic.
"""

from fontaine.config.schema import ModelConfig
from fontaine.models.transformer import FontaineModel


def build_model(config: ModelConfig, mesh: object | None = None) -> FontaineModel:
    """Build a model instance from validated configuration.

    ``mesh`` is a :class:`~fontaine.distributed.topology.ParallelTopology` for
    this rank. ``None`` builds every layer on one device, which is the
    single-process path.
    """
    if config.architecture in ("decoder_transformer", "moe_decoder"):
        return FontaineModel(config, mesh=mesh)
    raise ValueError(
        f"unknown architecture {config.architecture!r} — register it in "
        f"fontaine.models.factory to extend Fontaine"
    )
