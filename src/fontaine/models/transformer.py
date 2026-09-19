"""The Fontaine decoder-only Transformer language model.

A single ``FontaineModel`` class serves every size from Fontaine Tiny to
Fontaine XL — the architecture lives entirely in ``ModelConfig``. The model
forward signature is the stable interface the rest of the system programs
against:

    model(input_ids, targets=None, cache=None) -> ModelOutput(logits, loss)

``targets`` uses ``-100`` as the ignore index so future SFT/prompt-masking
can hide prompt tokens from the loss without touching the model.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from fontaine.config.schema import ModelConfig
from fontaine.models.components import (
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    build_norm,
    build_rope_cache,
)
from fontaine.models.kv_cache import KVCache


@dataclass
class ModelOutput:
    """Forward-pass result. ``loss`` is None when ``targets`` is not given."""

    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    loss: torch.Tensor | None = None


class FontaineModel(nn.Module):
    """Decoder-only Transformer with RoPE, GQA, and optional KV-cache decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = False

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.final_norm = build_norm(config.hidden_size, config.normalization, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Tying the output head to the input embedding saves
            # vocab_size * hidden_size parameters — significant at small scale.
            self.lm_head.weight = self.token_embedding.weight

        # RoPE tables are derived state (not saved in checkpoints).
        cos, sin = build_rope_cache(
            config.max_sequence_length, config.head_dim, config.rope_theta, device="cpu"
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down residual-branch output projections (GPT-2 style) so deep
        # stacks start near identity and avoid early activation blow-up.
        for block in self.blocks:
            out_projs: list[nn.Linear] = [block.attn.out_proj]
            if hasattr(block.mlp, "down_proj"):
                out_projs.append(block.mlp.down_proj)  # type: ignore[arg-type]
            else:
                out_projs.append(block.mlp.proj)  # type: ignore[arg-type]
            for proj in out_projs:
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 / (2 * config.num_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self) -> None:
        """Trade ~30% compute for large activation-memory savings (training only)."""
        self.gradient_checkpointing = True

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Parameter count; ``non_embedding`` excludes the (tied) token embedding."""
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        embedding = self.config.vocab_size * self.config.hidden_size
        return total - embedding

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len] int64
        targets: torch.Tensor | None = None,  # [batch, seq_len], -100 = ignore
        cache: KVCache | None = None,
    ) -> ModelOutput:
        if cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("gradient checkpointing cannot be combined with a KV cache")
        batch, seq_len = input_ids.shape
        pos_offset = cache.pos if cache is not None else 0
        if pos_offset + seq_len > self.config.max_sequence_length:
            raise ValueError(
                f"sequence window [{pos_offset}, {pos_offset + seq_len}) exceeds "
                f"max_sequence_length={self.config.max_sequence_length}"
            )
        # Full RoPE tables; attention layers slice out [pos_offset, pos_offset+T)
        # using the cache position, so offset handling lives in exactly one place.
        rope = (self.rope_cos, self.rope_sin)

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(block, x, rope, cache, use_reentrant=False)
            else:
                x = block(x, rope, cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
        return ModelOutput(logits=logits, loss=loss)


# Re-exported so downstream code can import attention/FFN pieces for research
# experiments without digging into internals.
__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "FontaineModel",
    "KVCache",
    "ModelOutput",
    "TransformerBlock",
]
