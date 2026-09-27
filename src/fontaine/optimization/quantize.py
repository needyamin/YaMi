"""Inference precision: pick a weight format and convert a model for serving.

``auto`` chooses per device and model size:

- CPU, 20M parameters or more: int8 dynamic quantization of every
  ``nn.Linear`` (weights stored as int8, activations quantized per batch).
  This quarters Linear weight memory and speeds up matmuls on AVX2/VNNI CPUs.
- CPU, smaller models: fp32 (quantization overhead outweighs the gain).
- CUDA: bf16 when supported, fp32 otherwise.

Explicit ``bf16`` on a CPU without native bf16 (AVX-512 BF16 or AMX) falls
back to fp32 with a warning. ``int8`` is CPU-only.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from fontaine.utils.logging import get_logger

logger = get_logger("optimization")

INT8_AUTO_MIN_PARAMETERS = 20_000_000
QUANTIZATION_LEVELS = {"fp32": "F32", "bf16": "BF16", "int8": "Q8_0"}


def cpu_supports_bf16() -> bool:
    """True when this CPU has native bf16 matmul (AVX-512 BF16 or AMX)."""
    probes = ("_is_avx512_bf16_supported", "_is_amx_tile_supported")
    for name in probes:
        probe = getattr(torch.cpu, name, None)
        if probe is None:
            continue
        try:
            if probe():
                return True
        except RuntimeError:
            continue
    return False


def resolve_inference_precision(requested: str, device: torch.device, num_parameters: int) -> str:
    """Turn ``inference.precision`` into ``fp32``, ``bf16``, or ``int8`` for a device."""
    if requested not in ("auto", "fp32", "bf16", "int8"):
        raise ValueError(f"unknown inference precision {requested!r}")
    if requested == "fp32":
        return "fp32"
    if device.type == "cpu":
        if requested == "auto":
            return "int8" if num_parameters >= INT8_AUTO_MIN_PARAMETERS else "fp32"
        if requested == "bf16" and not cpu_supports_bf16():
            logger.warning("this CPU has no native bf16; running fp32 instead")
            return "fp32"
        return requested
    if requested == "int8":
        raise ValueError("int8 inference is CPU-only; use bf16 or fp32 on this device")
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        if requested == "bf16":
            logger.warning("this GPU has no bf16 support; running fp32 instead")
        return "fp32"
    if requested == "bf16":
        logger.warning("bf16 is not enabled for device %s; running fp32", device)
    return "fp32"


class Int8Linear(nn.Module):
    """Weight-only int8 Linear with one scale per output row.

    Fallback for PyTorch builds without dynamic quantization. Weights are
    stored as int8 and dequantized per call, so memory drops by 4x while the
    matmul itself runs in the activation dtype.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        self.register_buffer("weight_int8", torch.round(weight / scale).to(torch.int8))
        self.register_buffer("scale", scale)
        bias = linear.bias.detach().float().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def _replace_linears(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
        else:
            _replace_linears(child)


def quantize_int8(model: nn.Module, *, use_torch: bool = True) -> nn.Module:
    """Convert every ``nn.Linear`` in an eval-mode model to int8 weights."""
    model.eval()
    quantize_dynamic = getattr(getattr(torch, "ao", None), "quantization", None)
    quantize_dynamic = getattr(quantize_dynamic, "quantize_dynamic", None)
    if use_torch and quantize_dynamic is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except (RuntimeError, NotImplementedError) as exc:
            logger.warning("torch dynamic quantization failed (%s); using weight-only int8", exc)
    _replace_linears(model)
    return model


def apply_inference_precision(model: nn.Module, precision: str) -> nn.Module:
    """Return ``model`` converted to a resolved precision (``fp32``/``bf16``/``int8``)."""
    if precision == "fp32":
        return model.float()
    if precision == "bf16":
        return model.to(torch.bfloat16)
    if precision == "int8":
        return quantize_int8(model.float())
    raise ValueError(f"unresolved inference precision {precision!r}")


def weight_bytes(model: nn.Module) -> int:
    """Bytes held by (deduplicated) parameters, before any quantization."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def estimate_loaded_bytes(model: nn.Module, precision: str) -> int:
    """Weight bytes after ``apply_inference_precision`` (call on the fp32 model).

    int8 stores each Linear weight as one byte per value. A Linear tied to the
    embedding gets its own int8 copy while the embedding stays fp32.
    """
    full = weight_bytes(model)
    if precision == "bf16":
        return full // 2
    if precision != "int8":
        return full
    linear_values = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    shared = {
        p.data_ptr()
        for m in model.modules()
        if not isinstance(m, nn.Linear)
        for p in m.parameters(recurse=False)
    }
    replaced = sum(
        m.weight.numel() * 4
        for m in model.modules()
        if isinstance(m, nn.Linear) and m.weight.data_ptr() not in shared
    )
    return full - replaced + linear_values


def set_num_threads(num_threads: int) -> int:
    """Apply a CPU thread count (0 keeps PyTorch's default); return the active count."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()
