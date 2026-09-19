"""Memory estimator and precision resolution tests."""

import pytest
import torch

from fontaine.config.schema import ModelConfig
from fontaine.optimization import (
    estimate_parameter_count,
    estimate_training_memory,
    format_bytes,
    resolve_device,
    resolve_precision,
)


def _config(**overrides) -> ModelConfig:
    values = dict(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=384,
        max_sequence_length=64,
        tie_word_embeddings=True,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_parameter_count_analytic_matches_instantiated():
    config = _config()
    from fontaine.models import build_model

    model = build_model(config)
    assert model.num_parameters() == estimate_parameter_count(config)["total"]
    assert model.num_parameters(non_embedding=True) == estimate_parameter_count(config)["non_embedding"]


def test_optimizer_states_dominate_over_parameters():
    estimate = estimate_training_memory(_config(), batch_size=2, sequence_length=32)
    params = estimate_parameter_count(_config())["total"]
    assert estimate.parameters_bytes == 4 * params
    assert estimate.gradients_bytes == 4 * params
    assert estimate.optimizer_bytes == 8 * params  # AdamW keeps two moments
    # The "16 bytes/param before activations" rule of thumb must be visible.
    assert estimate.total_bytes > 16 * params


def test_estimate_scales_with_batch_and_layers():
    small = estimate_training_memory(_config(), batch_size=1, sequence_length=32)
    big_batch = estimate_training_memory(_config(), batch_size=8, sequence_length=32)
    big_model = estimate_training_memory(_config(num_layers=12), batch_size=1, sequence_length=32)
    assert big_batch.activations_bytes > small.activations_bytes
    assert big_model.activations_bytes > small.activations_bytes


def test_gradient_checkpointing_reduces_activations():
    full = estimate_training_memory(_config(), batch_size=8, sequence_length=64)
    ckpt = estimate_training_memory(
        _config(), batch_size=8, sequence_length=64, gradient_checkpointing=True
    )
    assert ckpt.activations_bytes < full.activations_bytes


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        estimate_training_memory(_config(), batch_size=1, sequence_length=8, optimizer="adafactor")


def test_format_bytes():
    assert format_bytes(512) == "512.00 B"
    assert format_bytes(1024**2).endswith("MiB")
    assert format_bytes(1024**4).endswith("TiB")


def test_resolve_precision_cpu_is_fp32():
    plan = resolve_precision("auto", torch.device("cpu"))
    assert plan.autocast_dtype is None and not plan.use_scaler


def test_resolve_device_invalid():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda") if not torch.cuda.is_available() else None
