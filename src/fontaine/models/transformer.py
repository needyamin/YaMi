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
    MixtureOfExperts,
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

    def __init__(self, config: ModelConfig, mesh: object | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.mesh = mesh
        self.gradient_checkpointing = False
        pipeline_parallel, pipeline_rank, expert_parallel, expert_rank = _mesh_placement(mesh)
        from fontaine.distributed.topology import layer_stage

        self.layer_ids = list(
            range(config.num_layers)
            if pipeline_parallel == 1
            else layer_stage(config.num_layers, pipeline_parallel, pipeline_rank)
        )
        self.owns_embedding = pipeline_rank == 0
        self.owns_head = pipeline_rank == pipeline_parallel - 1

        self.token_embedding = (
            nn.Embedding(config.vocab_size, config.hidden_size) if self.owns_embedding else None
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_idx, expert_parallel, expert_rank)
            for layer_idx in self.layer_ids
        )
        sequence_parallel = int(getattr(mesh, "sequence_parallel", 1) or 1)
        sequence_rank = int(getattr(mesh, "tensor_rank", 0) or 0)
        for block in self.blocks:
            block.sequence_parallel = sequence_parallel
            block.sequence_rank = sequence_rank
        self.final_norm = (
            build_norm(config.hidden_size, config.normalization, config.norm_eps, config.kernel)
            if self.owns_head
            else None
        )
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) if self.owns_head else None
        )
        if config.tie_word_embeddings and self.lm_head is not None and self.token_embedding is not None:
            # Tying the output head to the input embedding saves
            # vocab_size * hidden_size parameters — significant at small scale.
            self.lm_head.weight = self.token_embedding.weight
        elif config.tie_word_embeddings and pipeline_parallel > 1:
            raise ValueError(
                "tied embeddings need the embedding and the output head on the same "
                "pipeline stage. Set tie_word_embeddings=false when pipeline_parallel_size > 1, "
                "or keep pipeline_parallel_size at 1."
            )

        # RoPE tables are derived state (not saved in checkpoints).
        cos, sin = build_rope_cache(
            config.max_sequence_length,
            config.head_dim,
            config.rope_theta,
            device="cpu",
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down residual-branch output projections (GPT-2 style) so deep
        # stacks start near identity and avoid early activation blow-up.
        residual_std = 0.02 / (2 * config.num_layers) ** 0.5
        for block in self.blocks:
            for proj in _residual_projections(block):
                nn.init.normal_(proj.weight, mean=0.0, std=residual_std)

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

    def num_active_parameters(self) -> int:
        """Weights used for one token.

        Dense models use every parameter. A mixture of experts leaves
        ``num_experts - num_experts_per_token`` experts idle per layer; those
        weights are stored (and trained, when selected) but not multiplied on
        a given token.
        """
        total = self.num_parameters()
        sharded = self.mesh is not None and (
            int(getattr(self.mesh, "pipeline_parallel", 1)) > 1
            or int(getattr(self.mesh, "expert_parallel", 1)) > 1
        )
        if sharded:
            from fontaine.optimization.memory import estimate_parameter_count

            return estimate_parameter_count(self.config)["active"]
        if self.config.num_experts <= 1:
            return total
        mlp = self.blocks[0].mlp
        if not isinstance(mlp, MixtureOfExperts):
            return total
        expert_params = sum(p.numel() for p in mlp.experts[0].parameters())
        idle = (self.config.num_experts - self.config.num_experts_per_token) * expert_params
        return total - self.config.num_layers * idle

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len] int64
        targets: torch.Tensor | None = None,  # [batch, seq_len], -100 = ignore
        cache: KVCache | None = None,
        document_ids: torch.Tensor | None = None,
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

        if self.mesh is not None and getattr(self.mesh, "pipeline_parallel", 1) > 1:
            return self._pipeline_forward(input_ids, targets, cache, document_ids, rope)

        x = self.token_embedding(input_ids)
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if document_ids is None:
                    x, aux = torch_checkpoint(block, x, rope, cache, use_reentrant=False)
                else:
                    x, aux = torch_checkpoint(
                        block, x, rope, cache, document_ids, use_reentrant=False
                    )
            else:
                x, aux = block(x, rope, cache, document_ids)
            aux_losses.append(aux)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
            # Load-balance loss is a training signal only. Eval and generation
            # report plain cross-entropy so validation stays comparable.
            if (
                self.training
                and self.config.num_experts > 1
                and self.config.moe_aux_loss_coef > 0
                and self.config.load_balancing_enabled
            ):
                aux_total = torch.stack([aux.reshape(()) for aux in aux_losses]).sum()
                loss = loss + self.config.moe_aux_loss_coef * aux_total
        return ModelOutput(logits=logits, loss=loss)

    def routing_stats(self) -> list[dict[str, float]]:
        """Per-layer router statistics from the most recent forward, if this model has experts."""
        stats = []
        for block in self.blocks:
            mlp = block.mlp
            if isinstance(mlp, MixtureOfExperts) and mlp.last_stats:
                stats.append(dict(mlp.last_stats))
        return stats

    def _pipeline_forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        cache: KVCache | None,
        document_ids: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> ModelOutput:
        import torch.distributed as dist

        if cache is not None:
            raise ValueError(
                "pipeline parallel does not use a KV cache. "
                "Run inference with pipeline_parallel_size=1."
            )
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "pipeline_parallel_size > 1 needs an initialized torch.distributed "
                "process group. Launch with torchrun. Training was not started."
            )
        hidden = self.config.hidden_size
        prev_rank = self.mesh.pipeline_prev()
        next_rank = self.mesh.pipeline_next()
        if self.owns_embedding:
            boundary = self.token_embedding(input_ids)
        else:
            if prev_rank is None:
                raise RuntimeError("pipeline stage is missing its previous rank")
            boundary = torch.empty(
                input_ids.shape[0], input_ids.shape[1], hidden, device=input_ids.device
            )
            dist.recv(boundary, src=prev_rank)
            if self.training and targets is not None:
                boundary.requires_grad_(True)
        x = boundary
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux = block(x, rope, None, document_ids)
            aux_losses.append(aux)
        aux_term = self._pipeline_aux(aux_losses)
        if not self.owns_head:
            if next_rank is None:
                raise RuntimeError("pipeline stage is missing its next rank")
            dist.send(x.contiguous(), dst=next_rank)
            if self.training and targets is not None:
                grad = torch.empty_like(x)
                dist.recv(grad, src=next_rank)
                surrogate = (x * grad.detach()).sum()
                if aux_term is not None:
                    surrogate = surrogate + aux_term
                surrogate.backward()
                if prev_rank is not None:
                    if boundary.grad is None:
                        raise RuntimeError(
                            "pipeline stage produced no gradient for the activation "
                            "it received. The stage graph is disconnected."
                        )
                    dist.send(boundary.grad.contiguous(), dst=prev_rank)
            empty = x.new_zeros(input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)
            return ModelOutput(logits=empty, loss=empty.sum().detach())
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
            if aux_term is not None:
                loss = loss + aux_term
            if self.training and prev_rank is not None:
                loss.backward()
                if boundary.grad is None:
                    raise RuntimeError(
                        "pipeline head produced no gradient for the activation it received."
                    )
                dist.send(boundary.grad.contiguous(), dst=prev_rank)
                return ModelOutput(logits=logits.detach(), loss=loss.detach())
        return ModelOutput(logits=logits, loss=loss)

    def _pipeline_aux(self, aux_losses: list[torch.Tensor]) -> torch.Tensor | None:
        if not (
            self.training
            and self.config.num_experts > 1
            and self.config.moe_aux_loss_coef > 0
            and self.config.load_balancing_enabled
            and aux_losses
        ):
            return None
        return self.config.moe_aux_loss_coef * torch.stack(
            [aux.reshape(()) for aux in aux_losses]
        ).sum()


def _mesh_placement(mesh: object | None) -> tuple[int, int, int, int]:
    """Return pipeline size, pipeline rank, expert size, expert rank."""
    if mesh is None:
        return 1, 0, 1, 0
    return (
        int(getattr(mesh, "pipeline_parallel", 1)),
        int(getattr(mesh, "pipeline_rank", 0)),
        int(getattr(mesh, "expert_parallel", 1)),
        int(getattr(mesh, "expert_rank", 0)),
    )


def _residual_projections(block: TransformerBlock) -> list[nn.Linear]:
    """Output projections whose init is scaled down with depth."""
    projections = [block.attn.out_proj]
    mlp = block.mlp
    if isinstance(mlp, MixtureOfExperts):
        for expert in list(mlp.experts) + list(mlp.shared_experts):
            projections.append(expert.down_proj if hasattr(expert, "down_proj") else expert.proj)
    elif hasattr(mlp, "down_proj"):
        projections.append(mlp.down_proj)
    else:
        projections.append(mlp.proj)
    return projections


# Re-exported so downstream code can import attention/FFN pieces for research
# experiments without digging into internals.
__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "MixtureOfExperts",
    "FontaineModel",
    "KVCache",
    "ModelOutput",
    "TransformerBlock",
]
