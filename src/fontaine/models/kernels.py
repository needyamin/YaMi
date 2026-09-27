"""Kernel selection for normalization, activations, and matrix multiplies.

``auto`` keeps the implementations the rest of YaMi is tested against.
``reference`` is the explicit formula. ``optimized`` calls the corresponding
PyTorch kernel and fails if that kernel is not in this build.
"""

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, mode: str) -> torch.Tensor:
    if mode == "optimized":
        if not hasattr(F, "rms_norm"):
            raise RuntimeError(
                "kernel=optimized requested torch.nn.functional.rms_norm, "
                "which this PyTorch build does not provide. "
                "Use kernel=reference or kernel=auto."
            )
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    x_float = x.float()
    normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(dtype=x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "reference":
        return gate.float().mul(torch.sigmoid(gate.float())).to(dtype=gate.dtype) * up
    return F.silu(gate) * up


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mode: str) -> torch.Tensor:
    if mode == "reference":
        y = torch.matmul(x, weight.transpose(-1, -2))
        if bias is not None:
            y = y + bias
        return y
    return F.linear(x, weight, bias)
