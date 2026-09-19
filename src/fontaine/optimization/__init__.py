"""Memory and precision tooling for resource-constrained training."""

from fontaine.optimization.memory import (
    MemoryEstimate,
    estimate_parameter_count,
    estimate_training_memory,
    format_bytes,
)
from fontaine.optimization.precision import (
    PrecisionPlan,
    resolve_device,
    resolve_precision,
)

__all__ = [
    "MemoryEstimate",
    "PrecisionPlan",
    "estimate_parameter_count",
    "estimate_training_memory",
    "format_bytes",
    "resolve_device",
    "resolve_precision",
]
