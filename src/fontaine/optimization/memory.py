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


def _norm_params(dim: int, kind: str) -> int:
    """Trainable weights in one norm. LayerNorm stores a bias; RMSNorm does not."""
    if kind == "layernorm":
        return 2 * dim
    return dim


def _ffn_params(hidden: int, intermediate: int, activation: str, bias: bool) -> int:
    if activation == "swiglu":
        count = 3 * hidden * intermediate
        if bias:
            count += 2 * intermediate + hidden
        return count
    count = 2 * hidden * intermediate
    if bias:
        count += intermediate + hidden
    return count


def estimate_parameter_count(config: ModelConfig) -> dict[str, int]:
    """Analytic parameter count (no model instantiation, works on any size).

    ``total`` counts every stored weight, including idle experts. ``active``
    counts the weights touched for one token: attention, norms, the router,
    every shared expert, and ``num_experts_per_token`` routed experts. A dense
    model has ``active == total``. Training memory uses ``total`` because
    AdamW stores a state for every expert.
    """
    if isinstance(config.vocab_size, str):
        raise ValueError(
            "estimate_parameter_count needs a numeric vocab_size; "
            "resolve model.vocab_size=auto from a tokenizer first"
        )
    h = config.hidden_size
    head_dim = config.head_dim
    q_out = config.num_attention_heads * head_dim
    kv_dim = config.num_kv_heads * head_dim

    q = h * q_out + (q_out if config.attention_bias else 0)
    k = h * kv_dim + (kv_dim if config.attention_bias else 0)
    v = k
    o = q_out * h + (h if config.attention_bias else 0)
    attention = q + k + v + o

    qk = 2 * head_dim if config.qk_norm else 0
    block_norms = _norm_params(h, config.normalization) * 2 + qk
    final_norm = _norm_params(h, config.normalization)
    normalization = config.num_layers * block_norms + final_norm

    dense_one = _ffn_params(h, config.intermediate_size, config.activation, config.mlp_bias)
    expert_one = _ffn_params(
        h, config.resolved_expert_intermediate(), config.activation, config.mlp_bias
    )
    if config.num_experts > 1:
        router = h * config.num_experts  # bias-free router
        moe_experts = config.num_experts * expert_one
        shared = config.num_shared_experts * expert_one
        dense_ffn = 0
        ffn_total = moe_experts + shared + router
        ffn_active = config.num_experts_per_token * expert_one + shared + router
    else:
        router = 0
        moe_experts = 0
        shared = 0
        dense_ffn = dense_one
        ffn_total = dense_ffn
        ffn_active = dense_ffn

    per_layer_total = attention + ffn_total + block_norms
    per_layer_active = attention + ffn_active + block_norms
    embedding = int(config.vocab_size) * h
    lm_head = 0 if config.tie_word_embeddings else embedding
    total = config.num_layers * per_layer_total + embedding + lm_head + final_norm
    active = config.num_layers * per_layer_active + embedding + lm_head + final_norm
    return {
        "embedding": embedding,
        "attention": config.num_layers * attention,
        "q": config.num_layers * q,
        "k": config.num_layers * k,
        "v": config.num_layers * v,
        "o": config.num_layers * o,
        "dense_ffn": config.num_layers * dense_ffn,
        "moe_experts": config.num_layers * moe_experts,
        "router": config.num_layers * router,
        "shared_experts": config.num_layers * shared,
        "normalization": normalization,
        "lm_head": lm_head,
        "total": total,
        "active": active,
        "non_embedding": total - embedding - lm_head,
    }


def estimate_inference_memory(
    config: ModelConfig, precision: str = "fp32", context_length: int | None = None
) -> dict[str, int]:
    """Bytes to serve one request: weights in ``precision`` plus a full KV cache.

    ``int8`` keeps embeddings and norms in fp32 and stores each Linear weight
    (including a tied output head's own copy) at one byte per value. ``int4``
    packs two weights per byte and keeps one fp32 scale per group. The KV
    cache runs in fp32, or bf16/fp16 for those compute dtypes.
    """
    counts = estimate_parameter_count(config)
    total = counts["total"]
    embedding = counts["embedding"]
    norms = counts["normalization"]
    if precision == "fp32":
        weights = 4 * total
    elif precision in ("bf16", "fp16"):
        weights = 2 * total
    elif precision == "int8":
        linear = total - embedding - norms
        if config.tie_word_embeddings:
            linear += embedding
        weights = 4 * (embedding + norms) + linear
    elif precision == "int4":
        from fontaine.optimization.quantize import estimate_int4_weight_bytes

        weights = estimate_int4_weight_bytes(config, counts)
    else:
        raise ValueError(f"unknown inference precision {precision!r}")
    window = context_length or config.max_sequence_length
    kv_bytes = 2 if precision in ("bf16", "fp16") else 4
    kv_cache = 2 * config.num_layers * config.num_kv_heads * config.head_dim * window * kv_bytes
    return {"weights": weights, "kv_cache": kv_cache, "total": weights + kv_cache}


def estimate_kv_cache_bytes(
    config: ModelConfig, context_length: int | None = None, bytes_per_element: int = 4
) -> int:
    """Bytes for one sequence of K and V at ``context_length`` (default: the model window)."""
    window = context_length or config.max_sequence_length
    return (
        2
        * config.num_layers
        * config.num_kv_heads
        * config.head_dim
        * window
        * bytes_per_element
    )


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


def suggest_micro_batch(
    config: ModelConfig,
    sequence_length: int,
    requested: int,
    device_bytes: int | None,
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> int:
    """Largest micro-batch at or below ``requested`` that fits ``device_bytes``.

    Raises when memory cannot be measured or even one sequence does not fit.
    The configured batch size is never kept silently in those cases.
    """
    if requested < 1:
        raise ValueError(f"batch_size must be >= 1, got {requested}")
    if device_bytes is None or device_bytes <= 0:
        raise RuntimeError(
            "training.fit_batch_to_memory is set, but available device memory could not "
            "be measured. Unset the flag or run where RAM or GPU memory is visible. "
            "The configured batch_size was not kept."
        )
    chosen = 0
    for size in range(1, requested + 1):
        estimate = estimate_training_memory(
            config,
            size,
            sequence_length,
            optimizer=optimizer,
            gradient_checkpointing=gradient_checkpointing,
        )
        if estimate.total_bytes <= device_bytes:
            chosen = size
        else:
            break
    if chosen < 1:
        needed = estimate_training_memory(
            config,
            1,
            sequence_length,
            optimizer=optimizer,
            gradient_checkpointing=gradient_checkpointing,
        )
        raise RuntimeError(
            "training.fit_batch_to_memory cannot fit a micro-batch of 1: "
            f"the estimate is {needed.total_bytes} bytes and the device has {device_bytes} bytes. "
            "Reduce the model or sequence length, or add memory. "
            "The configured batch_size was not kept."
        )
    return chosen
