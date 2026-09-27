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
QUANTIZATION_LEVELS = {"fp32": "F32", "bf16": "BF16", "fp16": "F16", "int8": "Q8_0", "int4": "Q4_0"}
INT4_GROUP = 32


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
    """Turn ``inference.precision`` into a concrete weight format for a device.

    ``auto`` never selects int4. Ask for ``int4`` explicitly.
    """
    if requested not in ("auto", "fp32", "bf16", "fp16", "int8", "int4"):
        raise ValueError(f"unknown inference precision {requested!r}")
    if requested in ("fp32", "fp16", "int4"):
        return requested
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
    """Return ``model`` converted to a resolved precision."""
    if precision == "fp32":
        return model.float()
    if precision == "bf16":
        return model.to(torch.bfloat16)
    if precision == "fp16":
        return model.half()
    if precision == "int8":
        return quantize_int8(model.float())
    if precision == "int4":
        return quantize_int4(model.float())
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
    if precision in ("bf16", "fp16"):
        return full // 2
    if precision == "int4":
        return _estimate_int4_module_bytes(model)
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


def int4_group_size(in_features: int, requested: int = INT4_GROUP) -> int:
    """Largest group size <= ``requested`` that divides ``in_features``.

    The requested size is used when it divides the row. Otherwise the largest
    divisor at or below ``requested`` is used, so every row is covered without
    padding the weight matrix.
    """
    if in_features < 1:
        raise ValueError(f"in_features must be >= 1, got {in_features}")
    if requested < 1:
        raise ValueError(f"int4 group size must be >= 1, got {requested}")
    limit = min(requested, in_features)
    for size in range(limit, 0, -1):
        if in_features % size == 0:
            return size
    return in_features


class Int4Linear(nn.Module):
    """Weight-only groupwise int4 Linear.

    Two int4 values share a byte. Each group along the input dimension has one
    fp32 scale. The matmul dequantizes to the activation dtype.
    """

    def __init__(self, linear: nn.Linear, group_size: int = INT4_GROUP) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        out, inn = weight.shape
        group = int4_group_size(inn, group_size)
        grouped = weight.view(out, inn // group, group)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        quantized = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
        flat = quantized.reshape(out, inn)
        if inn % 2:
            flat = F.pad(flat, (0, 1))
        lo = flat[:, 0::2].to(torch.uint8) & 0x0F
        hi = flat[:, 1::2].to(torch.uint8) & 0x0F
        self.register_buffer("packed", lo | (hi << 4))
        self.register_buffer("scale", scale.squeeze(-1).contiguous())
        bias = linear.bias.detach().float().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)
        self.in_features = inn
        self.out_features = out
        self.group_size = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.packed
        lo = (packed & 0x0F).to(torch.int8)
        hi = ((packed >> 4) & 0x0F).to(torch.int8)
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        flat = torch.stack([lo, hi], dim=-1).reshape(self.out_features, -1)
        flat = flat[:, : self.in_features]
        scale = self.scale.to(dtype=x.dtype)
        weight = flat.to(dtype=x.dtype) * scale.repeat_interleave(self.group_size, dim=1)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def _replace_linears_int4(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int4Linear(child))
        else:
            _replace_linears_int4(child)


def quantize_int4(model: nn.Module) -> nn.Module:
    """Convert every ``nn.Linear`` to groupwise int4 weights. The model must be in eval mode."""
    model.eval()
    _replace_linears_int4(model)
    return model


def _linear_int4_bytes(out_features: int, in_features: int, bias: bool) -> int:
    group = int4_group_size(in_features)
    packed_cols = (in_features + 1) // 2
    packed = out_features * packed_cols
    scales = out_features * (in_features // group) * 4
    bias_bytes = out_features * 4 if bias else 0
    return packed + scales + bias_bytes


def _estimate_int4_module_bytes(model: nn.Module) -> int:
    """Storage after ``quantize_int4``, counted on the still-fp32 module."""
    full = weight_bytes(model)
    shared = {
        p.data_ptr()
        for module in model.modules()
        if not isinstance(module, nn.Linear)
        for p in module.parameters(recurse=False)
    }
    replaced = 0
    quantized = 0
    seen: set[int] = set()
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        pointer = module.weight.data_ptr()
        quantized += _linear_int4_bytes(
            module.out_features, module.in_features, module.bias is not None
        )
        if pointer in shared or pointer in seen:
            continue
        seen.add(pointer)
        replaced += module.weight.numel() * module.weight.element_size()
        if module.bias is not None:
            replaced += module.bias.numel() * module.bias.element_size()
    return full - replaced + quantized


def estimate_int4_weight_bytes(config: object, counts: dict[str, int]) -> int:
    """Analytic int4 weight bytes from a model config (no module allocation)."""
    from fontaine.config.schema import ModelConfig

    if not isinstance(config, ModelConfig):
        raise TypeError("estimate_int4_weight_bytes expects a ModelConfig")
    h = config.hidden_size
    q_out = config.num_attention_heads * config.head_dim
    kv = config.num_kv_heads * config.head_dim
    bias = config.attention_bias
    mlp_bias = config.mlp_bias
    per_layer = (
        _linear_int4_bytes(q_out, h, bias)
        + _linear_int4_bytes(kv, h, bias)
        + _linear_int4_bytes(kv, h, bias)
        + _linear_int4_bytes(h, q_out, bias)
    )
    if config.num_experts > 1:
        width = config.resolved_expert_intermediate()
        copies = config.num_experts + config.num_shared_experts
        per_layer += _linear_int4_bytes(config.num_experts, h, False)
    else:
        width = config.intermediate_size
        copies = 1
    if config.activation == "swiglu":
        one = (
            _linear_int4_bytes(width, h, mlp_bias)
            + _linear_int4_bytes(width, h, mlp_bias)
            + _linear_int4_bytes(h, width, mlp_bias)
        )
    else:
        one = _linear_int4_bytes(width, h, mlp_bias) + _linear_int4_bytes(h, width, mlp_bias)
    per_layer += copies * one
    linear = config.num_layers * per_layer
    if not config.tie_word_embeddings:
        linear += _linear_int4_bytes(int(config.vocab_size), h, False)
    else:
        linear += _linear_int4_bytes(int(config.vocab_size), h, False)
    fp32 = 4 * (counts["embedding"] + counts["normalization"])
    return fp32 + linear
