"""Analytic memory estimation — plan runs *before* allocating them.

Training memory is dominated by four components (fp32 master weights):

- parameters:            ~4 bytes/param
- gradients:             ~4 bytes/param
- AdamW optimizer state: ~8 bytes/param (two moment buffers)
- activations:           scales with layers x batch x seq x hidden

So an AdamW training step costs roughly 16 bytes/param *plus* activations —
a 1B-parameter model needs ~19 GB before a single activation is stored.
The estimator makes this arithmetic explicit; see
``docs/training/memory.md`` for the full methodology and 16 GB feasibility
tables.
"""

from dataclasses import dataclass

from fontaine.config.schema import ModelConfig

_BYTES_PER_PARAM = 4  # fp32 master weights
_BYTES_PER_GRAD = 4
_BYTES_PER_OPTIMIZER_STATE = 4  # AdamW keeps two of these -> 8 bytes/param
_OPTIMIZER_STATES = {"adamw": 2, "adam": 2, "sgd": 0}
_ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN = 24  # rough fp32 constant, per layer


@dataclass
class MemoryEstimate:
    """Byte estimates for one training step (or inference when gradients=0)."""

    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    activations_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.parameters_bytes
            + self.gradients_bytes
            + self.optimizer_bytes
            + self.activations_bytes
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "parameters": format_bytes(self.parameters_bytes),
            "gradients": format_bytes(self.gradients_bytes),
            "optimizer_states": format_bytes(self.optimizer_bytes),
            "activations": format_bytes(self.activations_bytes),
            "total": format_bytes(self.total_bytes),
        }


def format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PiB"


def estimate_parameter_count(config: ModelConfig) -> dict[str, int]:
    """Analytic parameter count (no model instantiation, works on any size)."""
    h = config.hidden_size
    head_dim = config.head_dim
    kv_dim = config.num_kv_heads * head_dim

    if config.activation == "swiglu":
        ffn = 3 * h * config.intermediate_size
    else:
        ffn = 2 * h * config.intermediate_size
    per_layer = (h * h + 2 * h * kv_dim + h * h) + ffn + 2 * h  # attn + ffn + 2 norms
    embedding = config.vocab_size * h
    head = 0 if config.tie_word_embeddings else embedding
    total = config.num_layers * per_layer + embedding + head + h  # + final norm
    return {"total": total, "non_embedding": total - embedding - head}


def estimate_training_memory(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """Estimate steady-state training memory for a micro-batch step."""
    params = estimate_parameter_count(config)["total"]
    states = _OPTIMIZER_STATES.get(optimizer)
    if states is None:
        raise ValueError(f"unknown optimizer {optimizer!r} for memory estimation")
    optimizer_bytes = states * _BYTES_PER_OPTIMIZER_STATE * params

    tokens = batch_size * sequence_length
    activations = (
        config.num_layers
        * tokens
        * config.hidden_size
        * _ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN
    )
    if gradient_checkpointing:
        # Only ~1/num_layers of activations are kept; a sqrt approximation
        # is common because of recompute buffers.
        activations = max(activations // config.num_layers, tokens * config.hidden_size * 4)

    return MemoryEstimate(
        parameters_bytes=_BYTES_PER_PARAM * params,
        gradients_bytes=_BYTES_PER_GRAD * params,
        optimizer_bytes=optimizer_bytes,
        activations_bytes=activations,
    )
