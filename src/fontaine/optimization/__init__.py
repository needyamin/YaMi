"""Memory and precision tooling for resource-constrained training."""

from fontaine.optimization.memory import (
    MemoryEstimate,
    estimate_inference_memory,
    estimate_parameter_count,
    estimate_training_memory,
    format_bytes,
)
from fontaine.optimization.precision import (
    PrecisionPlan,
    resolve_device,
    resolve_precision,
)
from fontaine.optimization.quantize import (
    QUANTIZATION_LEVELS,
    apply_inference_precision,
    cpu_supports_bf16,
    quantize_int8,
    resolve_inference_precision,
    set_num_threads,
)

__all__ = [
    "QUANTIZATION_LEVELS",
    "MemoryEstimate",
    "PrecisionPlan",
    "apply_inference_precision",
    "cpu_supports_bf16",
    "estimate_inference_memory",
    "estimate_parameter_count",
    "estimate_training_memory",
    "format_bytes",
    "quantize_int8",
    "resolve_device",
    "resolve_inference_precision",
    "resolve_precision",
    "set_num_threads",
]
