"""Reusable Transformer building blocks.

Components are intentionally boring and standard: RMSNorm, rotary position
embeddings (RoPE), grouped-query attention (GQA), and a SwiGLU feed-forward
network. Each maps to a field of ``ModelConfig`` so architectures are
described by configuration, not code changes. Detailed rationale lives in
``docs/model/design.md``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from fontaine.config.schema import ModelConfig
from fontaine.models.attention import (
    apply_document_mask,
    attention_spec,
    build_position_mask,
    reference_attention,
    resolve_attention_backend,
)
from fontaine.models.kernels import linear, rms_norm, swiglu
from fontaine.models.kv_cache import KVCache


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Used by Llama-class models: cheaper than LayerNorm and equally stable in
    pre-norm residual stacks. Computed in float32 for numerical stability
    under autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-5, kernel: str = "auto") -> None:
        super().__init__()
        self.eps = eps
        self.kernel = kernel
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, self.kernel)


def build_norm(dim: int, kind: str, eps: float, kernel: str = "auto") -> nn.Module:
    """Factory for the configurable normalization layer."""
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps, kernel=kernel)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown normalization: {kind!r} (expected 'rmsnorm' or 'layernorm')")


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | str,
    scaling_type: str = "none",
    scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE: each has shape [seq_len, head_dim // 2].

    ``theta`` controls the wavelength base; larger theta (e.g. 1e6) extends
    useful context — the primary context-length scaling lever.

    ``scaling_type`` stretches a trained window: ``linear`` divides positions
    by ``scaling_factor`` (position interpolation); ``ntk`` raises the base by
    ``factor ** (d / (d - 2))`` so high-frequency dimensions stay intact.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    if scaling_type == "ntk" and scaling_factor > 1.0 and head_dim > 2:
        theta = theta * scaling_factor ** (head_dim / (head_dim - 2))
    elif scaling_type not in ("none", "linear", "ntk"):
        raise ValueError(f"unknown rope scaling type: {scaling_type!r}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    if scaling_type == "linear":
        positions = positions / scaling_factor
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim // 2]
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape [batch, heads, seq, head_dim].

    ``cos``/``sin`` are already position-sliced by the caller: [seq, head_dim//2].
    Uses the half-split convention (rotate_half), identical to GPT-NeoX/Llama.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _probe_native_gqa() -> bool:
    q = torch.zeros(1, 2, 1, 2)
    kv = torch.zeros(1, 1, 1, 2)
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except (TypeError, RuntimeError):
        return False
    return True


# SDPA reads shared K/V heads directly when torch supports ``enable_gqa``.
NATIVE_GQA = _probe_native_gqa()


def attention_mask(
    query_start: int, query_len: int, key_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Boolean [query_len, key_len] mask: causal, and within ``window`` when > 0."""
    query_pos = torch.arange(query_start, query_start + query_len, device=device)
    key_pos = torch.arange(key_len, device=device)
    allowed = key_pos[None, :] <= query_pos[:, None]
    if window > 0:
        allowed = allowed & (key_pos[None, :] > query_pos[:, None] - window)
    return allowed


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and grouped-query attention.

    GQA shares K/V across groups of query heads (``num_kv_heads`` divides
    ``num_attention_heads``): it cuts KV-cache memory and bandwidth roughly
    proportionally to the head ratio, which is what makes long-context and
    multi-GPU inference tractable. ``num_kv_heads == num_attention_heads``
    degenerates to standard MHA.

    Masking:
    - a query block starting at position 0 with full attention uses fused
      causal SDPA (no explicit mask);
    - one decode token with full attention sees the whole cache (no mask);
    - sliding-window layers and chunked prefill build a boolean mask.

    ``qk_norm`` normalizes each query/key head before RoPE. A layer's window
    comes from ``ModelConfig.layer_window``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.spec = attention_spec(config, layer_idx)
        self.window = self.spec.window if self.spec.kind == "sliding" else config.layer_window(layer_idx)
        self.backend_name = config.attention_backend
        self.kernel = config.kernel
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm: nn.Module | None = None
        self.k_norm: nn.Module | None = None
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps, kernel=config.kernel)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps, kernel=config.kernel)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        pos_offset = cache.pos if cache is not None else 0
        cos, sin = rope[0][pos_offset : pos_offset + seq_len], rope[1][pos_offset : pos_offset + seq_len]

        q = self._project(self.q_proj, x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._project(self.k_proj, x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self._project(self.v_proj, x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Store the shared K/V (num_kv_heads wide) BEFORE expanding heads for
        # GQA — the cache must hold only the compact shared representation.
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        grouped = self.num_kv_heads != self.num_heads
        sdpa_kwargs: dict[str, object] = {}
        if grouped and NATIVE_GQA:
            sdpa_kwargs["enable_gqa"] = True
        elif grouped:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        key_len = k.shape[2]
        query_end = pos_offset + seq_len
        dropout_p = self.dropout if self.training else 0.0
        backend = resolve_attention_backend(self.backend_name, q.device)
        if document_ids is not None and cache is not None:
            raise ValueError(
                "document-boundary masks apply to packed training sequences. "
                "Do not pass document_ids together with a KV cache."
            )
        windowed = self.spec.kind == "sparse" or (
            self.spec.kind == "sliding" and self.spec.window > 0 and query_end > self.spec.window
        )
        force_mask = document_ids is not None or backend == "reference" or self.spec.kind == "sparse"
        if force_mask or windowed or not (pos_offset == 0 or seq_len == 1):
            mask = build_position_mask(pos_offset, seq_len, key_len, self.spec, q.device)
            if document_ids is not None:
                mask = apply_document_mask(mask, document_ids)
            else:
                mask = mask[None, None, :, :]
        else:
            mask = None
        if backend == "reference":
            if grouped and k.shape[1] != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            y = reference_attention(q, k, v, mask, dropout_p, self.training)
        elif mask is None and pos_offset == 0:
            y = self._sdpa(q, k, v, dropout_p, True, backend, sdpa_kwargs, None)
        elif mask is None:
            y = self._sdpa(q, k, v, dropout_p, False, backend, sdpa_kwargs, None)
        else:
            y = self._sdpa(q, k, v, dropout_p, False, backend, sdpa_kwargs, mask)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self._project(self.out_proj, y)

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "reference":
            return linear(x, layer.weight, layer.bias, "reference")
        return layer(x)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        is_causal: bool,
        backend: str,
        sdpa_kwargs: dict[str, object],
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if backend == "flash" and mask is not None:
            raise RuntimeError(
                "attention backend flash does not implement sliding-window, sparse, "
                "or document-boundary masks. Use backend=optimized or reference."
            )

        def run() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal, **sdpa_kwargs
            )

        if backend != "flash":
            return run()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError as exc:
            raise RuntimeError(
                "attention backend flash needs torch.nn.attention.sdpa_kernel. "
                "Use backend=optimized or reference."
            ) from exc
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return run()
        except RuntimeError as exc:
            raise RuntimeError(
                "attention backend flash could not run this call "
                f"({exc}). Use backend=optimized or reference."
            ) from exc


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    ``swiglu`` (gated, used by Llama/most modern models) or classic GELU.
    The gated variant spends the parameter budget across three matrices
    (gate/up/down), so its intermediate size is typically ~2.7x hidden rather
    than the 4x used for GELU MLPs — the configs account for this.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.activation = config.activation
        self.kernel = config.kernel
        bias = config.mlp_bias
        if config.activation == "swiglu":
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        elif config.activation == "gelu":
            self.fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
            self.proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown activation: {config.activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate = self._project(self.gate_proj, x)
            up = self._project(self.up_proj, x)
            return self._project(self.down_proj, swiglu(gate, up, self.kernel))
        hidden = F.gelu(self._project(self.fc, x))
        return self._project(self.proj, hidden)

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "reference":
            return linear(x, layer.weight, layer.bias, "reference")
        return layer(x)


class MixtureOfExperts(nn.Module):
    """Token-choice mixture of experts replacing the dense feed-forward.

    A bias-free router scores every token. Only the top
    ``num_experts_per_token`` experts run, and only on the tokens that
    selected them — that is the active-parameter path. Expert outputs are
    mixed by the renormalized router weights.

    The auxiliary loss is the Switch load-balance term
    ``num_experts * sum(fraction_dispatched * mean_router_prob)``. The
    dispatch fraction is detached so the router is trained by the soft
    probabilities, not by the hard assignment.
    """

    def __init__(
        self,
        config: ModelConfig,
        expert_parallel_size: int = 1,
        expert_parallel_rank: int = 0,
    ) -> None:
        super().__init__()
        from fontaine.distributed.topology import expert_ids_for_rank

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.capacity_factor = config.expert_capacity_factor
        self.kernel = config.kernel
        self.expert_parallel_size = expert_parallel_size
        self.expert_parallel_rank = expert_parallel_rank
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.global_expert_ids = list(
            expert_ids_for_rank(config.num_experts, expert_parallel_size, expert_parallel_rank)
        )
        expert_config = _expert_feedforward_config(config)
        self.experts = nn.ModuleList(FeedForward(expert_config) for _ in self.global_expert_ids)
        self.shared_experts = nn.ModuleList(
            FeedForward(expert_config) for _ in range(config.num_shared_experts)
        )
        self.parallel_group = None
        self.last_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined, aux = self.local_forward(x)
        if self.expert_parallel_size > 1:
            combined = _all_reduce_expert_outputs(combined, self.parallel_group)
        for shared in self.shared_experts:
            combined = combined + shared(x)
        return combined, aux

    def local_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = x.shape
        probs = F.softmax(self.router(x), dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_v = top_v / top_v.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if self.kernel == "reference" and self.capacity_factor == 0 and self.expert_parallel_size == 1:
            routed, aux = self._reference_dispatch(x, probs, top_v, top_i)
            counts = torch.bincount(top_i.reshape(-1), minlength=self.num_experts)
            self._record_stats(probs, counts, 0)
            return routed, aux

        flat_x = x.reshape(-1, hidden)
        flat_i = top_i.reshape(-1, self.top_k)
        n_tokens = flat_x.shape[0]

        # Sort the (token, expert) pairs by expert once, then run each expert on
        # one contiguous slice: a single host sync instead of one per expert.
        expert_ids = flat_i.reshape(-1)
        order = torch.argsort(expert_ids, stable=True)
        token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)[order]
        pair_weights = top_v.reshape(-1)[order].unsqueeze(-1).to(flat_x.dtype)
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        capacity = 0
        if self.capacity_factor > 0:
            capacity = max(
                1, math.ceil(n_tokens * self.top_k / self.num_experts * self.capacity_factor)
            )
        local = {expert_id: index for index, expert_id in enumerate(self.global_expert_ids)}

        combined = flat_x.new_zeros(flat_x.shape)
        dropped = 0
        start = 0
        for expert_id, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            tokens = token_ids[start : start + count]
            weights = pair_weights[start : start + count]
            start += count
            if expert_id not in local:
                continue
            if capacity and count > capacity:
                choice = torch.topk(weights.squeeze(-1), capacity).indices
                dropped += count - capacity
                tokens = tokens.index_select(0, choice)
                weights = weights.index_select(0, choice)
            expert = self.experts[local[expert_id]]
            out = expert(flat_x.index_select(0, tokens)) * weights
            combined = combined.index_add(0, tokens, out)

        routed = combined.view(batch, seq_len, hidden)
        self._record_stats(probs, counts, dropped)

        assignment = F.one_hot(flat_i, num_classes=self.num_experts).to(probs.dtype)
        fraction = assignment.sum(dim=(0, 1)) / (n_tokens * self.top_k)
        mean_prob = probs.reshape(-1, self.num_experts).mean(dim=0)
        aux = self.num_experts * (fraction.detach() * mean_prob).sum()
        return routed, aux

    def _reference_dispatch(
        self,
        x: torch.Tensor,
        probs: torch.Tensor,
        top_v: torch.Tensor,
        top_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = x.shape
        flat_x = x.reshape(-1, hidden)
        flat_i = top_i.reshape(-1, self.top_k)
        flat_v = top_v.reshape(-1, self.top_k)
        combined = flat_x.new_zeros(flat_x.shape)
        for token in range(flat_x.shape[0]):
            for slot in range(self.top_k):
                expert = self.experts[int(flat_i[token, slot])]
                combined[token] = combined[token] + flat_v[token, slot] * expert(flat_x[token][None])[0]
        routed = combined.view(batch, seq_len, hidden)
        n_tokens = flat_x.shape[0]
        assignment = F.one_hot(flat_i, num_classes=self.num_experts).to(probs.dtype)
        fraction = assignment.sum(dim=(0, 1)) / (n_tokens * self.top_k)
        mean_prob = probs.reshape(-1, self.num_experts).mean(dim=0)
        aux = self.num_experts * (fraction.detach() * mean_prob).sum()
        return routed, aux

    def _record_stats(self, probs: torch.Tensor, counts: torch.Tensor, dropped: int) -> None:
        mean_prob = probs.reshape(-1, self.num_experts).mean(dim=0)
        entropy = -(mean_prob * mean_prob.clamp_min(1e-9).log()).sum()
        used = (counts > 0).to(mean_prob.dtype).mean()
        mean_count = counts.to(mean_prob.dtype).mean().clamp_min(1e-9)
        imbalance = counts.to(mean_prob.dtype).max() / mean_count
        self.last_stats = {
            "expert_utilization": float(used),
            "dropped_tokens": float(dropped),
            "routing_entropy": float(entropy),
            "load_imbalance": float(imbalance),
            "tokens": float(counts.sum()),
        }


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + attn(norm(x)); x + mlp(norm(x)).

    Pre-norm (normalize before the sublayer) is standard for deep stacks:
    it keeps residual paths clean and trains stably without warm-start tricks.
    ``num_experts > 1`` swaps the dense MLP for :class:`MixtureOfExperts`.
    The forward return is ``(hidden, aux_loss)`` so gradient checkpointing
    keeps the load-balance term in the graph. Dense blocks return a zero aux.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        expert_parallel_size: int = 1,
        expert_parallel_rank: int = 0,
    ) -> None:
        super().__init__()
        self.attn_norm = build_norm(
            config.hidden_size, config.normalization, config.norm_eps, config.kernel
        )
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp_norm = build_norm(
            config.hidden_size, config.normalization, config.norm_eps, config.kernel
        )
        self.mlp: FeedForward | MixtureOfExperts
        if config.num_experts > 1:
            self.mlp = MixtureOfExperts(config, expert_parallel_size, expert_parallel_rank)
        else:
            self.mlp = FeedForward(config)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.sequence_parallel = 1
        self.sequence_rank = 0

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.resid_dropout(
            self.attn(self.attn_norm(x), rope, cache, document_ids=document_ids)
        )
        mlp_input = x
        if self.sequence_parallel > 1:
            mlp_input = _split_sequence(x, self.sequence_rank, self.sequence_parallel)
        if isinstance(self.mlp, MixtureOfExperts):
            y, aux = self.mlp(self.mlp_norm(mlp_input))
        else:
            y = self.mlp(self.mlp_norm(mlp_input))
            aux = y.new_zeros(())
        if self.sequence_parallel > 1:
            y = _gather_sequence(y, self.sequence_parallel)
            aux = _average_sequence_scalar(aux, self.sequence_parallel)
        x = x + self.resid_dropout(y)
        return x, aux


def _expert_feedforward_config(config: ModelConfig) -> ModelConfig:
    width = config.resolved_expert_intermediate()
    if width == config.intermediate_size:
        return config
    from dataclasses import replace

    return replace(config, intermediate_size=width)


def _all_reduce_expert_outputs(hidden: torch.Tensor, group: object | None = None) -> torch.Tensor:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "expert_parallel_size > 1 needs an initialized torch.distributed "
            "process group so expert outputs can be summed. Launch with torchrun. "
            "Training was not started."
        )
    return _SumReduce.apply(hidden.contiguous(), group)


def _split_sequence(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if x.shape[1] % world != 0:
        raise RuntimeError(
            f"sequence length {x.shape[1]} is not divisible by sequence_parallel_size {world}"
        )
    return x.chunk(world, dim=1)[rank]


def _gather_sequence(local: torch.Tensor, world: int) -> torch.Tensor:
    import torch.distributed as dist

    from fontaine.distributed.groups import tensor_group

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "sequence_parallel_size > 1 splits the feed-forward sequence across ranks "
            "and all-gathers the result. torch.distributed is not initialized. "
            "Launch with torchrun. Training was not started."
        )
    return _GatherSequence.apply(local.contiguous(), world, tensor_group())


def _average_sequence_scalar(value: torch.Tensor, world: int) -> torch.Tensor:
    from fontaine.distributed.groups import tensor_group

    return _AverageReduce.apply(value, world, tensor_group())


class _SumReduce(torch.autograd.Function):
    """Differentiable sum across the expert-parallel group."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
        import torch.distributed as dist

        ctx.group = group
        out = hidden.detach().clone()
        if group is None:
            dist.all_reduce(out)
        else:
            dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        import torch.distributed as dist

        reduced = grad.contiguous().clone()
        if ctx.group is None:
            dist.all_reduce(reduced)
        else:
            dist.all_reduce(reduced, group=ctx.group)
        return reduced, None


class _AverageReduce(torch.autograd.Function):
    """Differentiable mean of a scalar across the sequence-parallel group."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, world: int, group) -> torch.Tensor:  # type: ignore[override]
        import torch.distributed as dist

        ctx.group = group
        ctx.world = world
        reduced = value.detach().clone()
        if group is None:
            dist.all_reduce(reduced)
        else:
            dist.all_reduce(reduced, group=group)
        return reduced / world

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        import torch.distributed as dist

        reduced = grad.detach().clone()
        if ctx.group is None:
            dist.all_reduce(reduced)
        else:
            dist.all_reduce(reduced, group=ctx.group)
        return reduced / ctx.world, None, None


class _GatherSequence(torch.autograd.Function):
    """All-gather on the sequence dimension, differentiable per rank slice."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, world: int, group) -> torch.Tensor:  # type: ignore[override]
        import torch.distributed as dist

        ctx.world = world
        ctx.group = group
        ctx.rank = dist.get_rank(group) if group is not None else dist.get_rank()
        parts = [torch.empty_like(local) for _ in range(world)]
        if group is None:
            dist.all_gather(parts, local)
        else:
            dist.all_gather(parts, local, group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad.chunk(ctx.world, dim=1)[ctx.rank], None, None
