"""Reusable Transformer building blocks.

Components are intentionally boring and standard: RMSNorm, rotary position
embeddings (RoPE), grouped-query attention (GQA), and a SwiGLU feed-forward
network. Each maps to a field of ``ModelConfig`` so architectures are
described by configuration, not code changes. Detailed rationale lives in
``docs/model/design.md``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fontaine.config.schema import ModelConfig
from fontaine.models.kv_cache import KVCache


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Used by Llama-class models: cheaper than LayerNorm and equally stable in
    pre-norm residual stacks. Computed in float32 for numerical stability
    under autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def build_norm(dim: int, kind: str, eps: float) -> nn.Module:
    """Factory for the configurable normalization layer."""
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown normalization: {kind!r} (expected 'rmsnorm' or 'layernorm')")


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE: each has shape [seq_len, head_dim // 2].

    ``theta`` controls the wavelength base; larger theta (e.g. 1e6) extends
    useful context — the primary context-length scaling lever.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
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
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and grouped-query attention.

    GQA shares K/V across groups of query heads (``num_kv_heads`` divides
    ``num_attention_heads``): it cuts KV-cache memory and bandwidth roughly
    proportionally to the head ratio, which is what makes long-context and
    multi-GPU inference tractable. ``num_kv_heads == num_attention_heads``
    degenerates to standard MHA.

    Two execution paths:
    - no cache (training/prefill): fused causal SDPA, no explicit mask needed.
    - with cache (incremental decode): explicit causal mask against the
      cached prefix; ``cache.update`` is called per layer.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        pos_offset = cache.pos if cache is not None else 0
        cos, sin = rope[0][pos_offset : pos_offset + seq_len], rope[1][pos_offset : pos_offset + seq_len]

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Store the shared K/V (num_kv_heads wide) BEFORE expanding heads for
        # GQA — the cache must hold only the compact shared representation.
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        # Grouped-query attention: expand shared K/V heads to match query heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if cache is not None:
            # Query at absolute position pos_offset+i may attend keys 0..pos_offset+i.
            query_pos = torch.arange(pos_offset, pos_offset + seq_len, device=q.device)
            key_pos = torch.arange(k.shape[2], device=q.device)
            attn_mask = key_pos[None, :] <= query_pos[:, None]  # [T, cached+T]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(y)


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
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.proj(F.gelu(self.fc(x)))


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(config) for _ in range(config.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = x.shape
        probs = F.softmax(self.router(x), dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_v = top_v / top_v.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_x = x.reshape(-1, hidden)
        flat_i = top_i.reshape(-1, self.top_k)
        flat_v = top_v.reshape(-1, self.top_k)
        combined = flat_x.new_zeros(flat_x.shape)

        for expert_id, expert in enumerate(self.experts):
            mask = (flat_i == expert_id).any(dim=-1)
            if not bool(mask.any().item()):
                continue
            index = mask.nonzero(as_tuple=True)[0]
            selected = expert(flat_x.index_select(0, index))
            weights = (flat_v * (flat_i == expert_id).to(flat_v.dtype)).sum(dim=-1)
            weighted = selected * weights.index_select(0, index).unsqueeze(-1)
            combined = combined + combined.new_zeros(combined.shape).index_add(0, index, weighted)

        n_tokens = flat_x.shape[0]
        assignment = F.one_hot(flat_i, num_classes=self.num_experts).to(probs.dtype)
        fraction = assignment.sum(dim=(0, 1)) / (n_tokens * self.top_k)
        mean_prob = probs.reshape(-1, self.num_experts).mean(dim=0)
        aux = self.num_experts * (fraction.detach() * mean_prob).sum()
        return combined.view(batch, seq_len, hidden), aux


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + attn(norm(x)); x + mlp(norm(x)).

    Pre-norm (normalize before the sublayer) is standard for deep stacks:
    it keeps residual paths clean and trains stably without warm-start tricks.
    ``num_experts > 1`` swaps the dense MLP for :class:`MixtureOfExperts`.
    The forward return is ``(hidden, aux_loss)`` so gradient checkpointing
    keeps the load-balance term in the graph. Dense blocks return a zero aux.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = build_norm(config.hidden_size, config.normalization, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp_norm = build_norm(config.hidden_size, config.normalization, config.norm_eps)
        self.mlp: FeedForward | MixtureOfExperts
        if config.num_experts > 1:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = FeedForward(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), rope, cache))
        if isinstance(self.mlp, MixtureOfExperts):
            y, aux = self.mlp(self.mlp_norm(x))
            x = x + self.resid_dropout(y)
            return x, aux
        x = x + self.resid_dropout(self.mlp(self.mlp_norm(x)))
        return x, x.new_zeros(())
