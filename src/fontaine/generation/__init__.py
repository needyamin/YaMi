"""Fontaine generation subsystem (sampling strategies)."""

from fontaine.generation.sampling import (
    SamplingConfig,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_token,
)

__all__ = [
    "SamplingConfig",
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "sample_token",
]
