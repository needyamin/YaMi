"""Device and mixed-precision resolution.

``precision: auto`` picks the fastest numerically safe option per device:
CUDA prefers bf16 (no loss scaling needed), falls back to fp16 + GradScaler;
MPS uses fp16 autocast; CPU stays fp32 (autocast on CPU is slow at this scale).
"""

from dataclasses import dataclass

import torch

from fontaine.utils.logging import get_logger

logger = get_logger("optimization")


@dataclass
class PrecisionPlan:
    """How a run should execute forward/backward passes."""

    device: torch.device
    autocast_dtype: torch.dtype | None  # None -> run in fp32, no autocast
    use_scaler: bool  # fp16 requires dynamic loss scaling


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda' but CUDA is not available")
    return device


def resolve_precision(precision: str, device: torch.device) -> PrecisionPlan:
    """Turn the config value into a concrete autocast/scaler plan."""
    if precision not in ("auto", "fp32", "bf16", "fp16"):
        raise ValueError(f"unknown precision {precision!r}")

    if precision == "fp32":
        plan = PrecisionPlan(device, None, False)
    elif device.type == "cuda":
        if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            plan = PrecisionPlan(device, torch.bfloat16, False)
        elif precision in ("auto", "fp16"):
            plan = PrecisionPlan(device, torch.float16, True)
        else:
            plan = PrecisionPlan(device, None, False)
    elif device.type == "mps":
        plan = (
            PrecisionPlan(device, torch.float16, False)
            if precision in ("auto", "fp16")
            else PrecisionPlan(device, None, False)
        )
    else:  # cpu
        if precision in ("bf16", "fp16"):
            logger.warning(
                "%s autocast on CPU is not beneficial; running fp32", precision
            )
        plan = PrecisionPlan(device, None, False)

    logger.info(
        "precision plan: device=%s autocast=%s scaler=%s",
        plan.device,
        plan.autocast_dtype,
        plan.use_scaler,
    )
    return plan
