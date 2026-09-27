"""Attention masks and backends.

``optimized`` uses scaled-dot-product attention. ``reference`` is an explicit
matmul/softmax path for debugging. ``flash`` requests the CUDA flash kernel
and fails when that kernel is not available. ``auto`` selects ``optimized``.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fontaine.config.schema import ModelConfig


@dataclass(frozen=True)
class AttentionSpec:
    """How one layer is allowed to attend."""

    kind: str  # full | sliding | sparse
    window: int
    sparse_pattern: str
    block_size: int
    stride: int
    global_tokens: int


def attention_spec(config: ModelConfig, layer_idx: int) -> AttentionSpec:
    kind = config.layer_kind(layer_idx)
    if kind == "sliding":
        window = config.sliding_window
    elif kind == "sparse":
        window = config.sparse_window or config.sliding_window
    else:
        window = 0
    return AttentionSpec(
        kind=kind,
        window=window,
        sparse_pattern=config.sparse_pattern,
        block_size=config.sparse_block_size,
        stride=config.sparse_stride,
        global_tokens=config.sparse_global_tokens,
    )


def available_attention_backends(device: torch.device | None = None) -> list[str]:
    """Backends that can run on ``device`` right now."""
    names = ["reference", "optimized"]
    if device is not None and device.type == "cuda" and _flash_available():
        names.append("flash")
    return names


def _flash_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.backends.cuda.flash_sdp_enabled())
    except (AttributeError, RuntimeError):
        return False


def resolve_attention_backend(requested: str, device: torch.device) -> str:
    """Pick a backend or raise an error that names what is available."""
    available = available_attention_backends(device)
    if requested == "auto":
        return "optimized"
    if requested == "flash":
        if "flash" not in available:
            raise RuntimeError(
                "attention backend 'flash' is not available on "
                f"{device.type}. Available backends: {available}. "
                "Use backend=auto, optimized, or reference."
            )
        return "flash"
    if requested in ("optimized", "reference"):
        if requested not in available and requested != "reference":
            raise RuntimeError(
                f"attention backend {requested!r} is not available. "
                f"Available backends: {available}."
            )
        return requested
    raise RuntimeError(
        f"unknown attention backend {requested!r}. "
        f"Available backends: {available_attention_backends(device)}."
    )


def build_position_mask(
    query_start: int,
    query_len: int,
    key_len: int,
    spec: AttentionSpec,
    device: torch.device,
) -> torch.Tensor:
    """Boolean ``[query_len, key_len]`` mask. True means the key is visible."""
    query_pos = torch.arange(query_start, query_start + query_len, device=device)
    key_pos = torch.arange(key_len, device=device)
    allowed = key_pos[None, :] <= query_pos[:, None]
    if spec.kind == "full":
        return allowed
    if spec.kind == "sliding":
        return allowed & (key_pos[None, :] > query_pos[:, None] - spec.window)
    return allowed & _sparse_visible(query_pos, key_pos, spec)


def _sparse_visible(
    query_pos: torch.Tensor, key_pos: torch.Tensor, spec: AttentionSpec
) -> torch.Tensor:
    distance = query_pos[:, None] - key_pos[None, :]
    local = distance < spec.window
    pattern = spec.sparse_pattern
    if pattern == "local":
        return local
    if pattern == "block":
        return (query_pos[:, None] // spec.block_size) == (key_pos[None, :] // spec.block_size)
    strided = distance % spec.stride == 0
    glob = key_pos[None, :] < spec.global_tokens
    if pattern == "strided":
        return local | strided
    if pattern == "global-local":
        return local | glob
    if pattern == "hybrid":
        return local | strided | glob
    raise RuntimeError(f"unknown sparse pattern {pattern!r}")


def apply_document_mask(
    allowed: torch.Tensor, document_ids: torch.Tensor
) -> torch.Tensor:
    """Restrict ``allowed`` so queries cannot see keys from another document.

    ``document_ids`` is ``[batch, key_len]`` for a full sequence, or
    ``[batch, query_len]`` aligned with the query positions already sliced by
    the caller. The mask becomes ``[batch, 1, query, key]``.
    """
    same = document_ids[:, :, None] == document_ids[:, None, :]
    return allowed[None, None, :, :] & same[:, None, :, :]


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Explicit scaled-dot-product attention. ``query`` is ``[batch, heads, seq, dim]``."""
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    if dropout_p and training:
        probs = F.dropout(probs, dropout_p)
    return torch.matmul(probs, value.float()).to(dtype=query.dtype)
