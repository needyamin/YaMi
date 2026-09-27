"""Describe a configuration so a checkpoint can be identified without loading weights."""

from fontaine.config.schema import ModelConfig
from fontaine.optimization.memory import estimate_parameter_count


def registry_record(config: ModelConfig, precision: str = "unspecified") -> dict[str, object]:
    counts = None
    if not isinstance(config.vocab_size, str):
        counts = estimate_parameter_count(config)
    return {
        "architecture": config.architecture,
        "version": config.architecture_version,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "attention": config.attention_type,
        "attention_backend": config.attention_backend,
        "moe": config.num_experts > 1,
        "num_experts": config.num_experts,
        "num_experts_per_token": config.num_experts_per_token,
        "num_shared_experts": config.num_shared_experts,
        "context_length": config.max_sequence_length,
        "precision": precision,
        "parameters": None if counts is None else counts["total"],
        "active_parameters": None if counts is None else counts["active"],
        "modalities": list(config.modalities),
    }
