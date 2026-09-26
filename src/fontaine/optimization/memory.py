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
    """Analytic parameter count (no model instantiation, works on any size).

    ``total`` counts every stored weight, including idle experts. ``active``
    counts the weights touched for one token: attention, norms, the router,
    and ``num_experts_per_token`` experts. A dense model (``num_experts == 1``)
    has ``active == total``. Training memory uses ``total`` because AdamW
    stores a state for every expert.
    """
    if isinstance(config.vocab_size, str):
        raise ValueError(
            "estimate_parameter_count needs a numeric vocab_size; "
            "resolve model.vocab_size=auto from a tokenizer first"
        )
    h = config.hidden_size
    kv_dim = config.num_kv_heads * config.head_dim

    attn = h * h + 2 * h * kv_dim + h * h
    if config.attention_bias:
        attn += h + kv_dim + kv_dim + h

    inter = config.intermediate_size
    if config.activation == "swiglu":
        one_expert = 3 * h * inter
        if config.mlp_bias:
            one_expert += 2 * inter + h
    else:
        one_expert = 2 * h * inter
        if config.mlp_bias:
            one_expert += inter + h

    if config.num_experts > 1:
        router = h * config.num_experts  # bias-free router
        ffn_total = config.num_experts * one_expert + router
        ffn_active = config.num_experts_per_token * one_expert + router
    else:
        ffn_total = one_expert
        ffn_active = one_expert

    norms = 2 * h  # two pre-norm RMSNorm weights
    per_layer_total = attn + ffn_total + norms
    per_layer_active = attn + ffn_active + norms
    embedding = config.vocab_size * h
    head = 0 if config.tie_word_embeddings else embedding
    final_norm = h
    total = config.num_layers * per_layer_total + embedding + head + final_norm
    active = config.num_layers * per_layer_active + embedding + head + final_norm
    return {
        "total": total,
        "active": active,
        "non_embedding": total - embedding - head,
    }


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
