"""Tests for model components: norms, RoPE, attention, causality."""

import pytest
import torch

from fontaine.config.schema import ModelConfig
from fontaine.models import KVCache
from fontaine.models.components import (
    RMSNorm,
    apply_rope,
    build_rope_cache,
)


def test_rmsnorm_shape_and_scale():
    norm = RMSNorm(16, eps=1e-5)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # RMS norm (weight=1) normalizes each vector to unit RMS.
    rms_out = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-4)


def test_rope_preserves_vector_norm():
    cos, sin = build_rope_cache(8, 16, theta=10000.0, device="cpu")
    x = torch.randn(1, 1, 8, 16)
    rotated = apply_rope(x, cos, sin)
    # Rotation must not change vector magnitudes.
    assert torch.allclose(
        x.pow(2).sum(-1).sqrt(), rotated.pow(2).sum(-1).sqrt(), atol=1e-5
    )


def test_rope_is_relative():
    """Same relative distance -> same similarity, regardless of absolute position."""
    cos, sin = build_rope_cache(32, 16, theta=10000.0, device="cpu")
    q = torch.randn(1, 1, 1, 16)
    # Rotate q as if at position 3 and k as if at position 7, vs 10 and 14.
    cos3, sin3 = build_rope_cache(32, 16, theta=10000.0, device="cpu")
    q_at_3 = apply_rope(q, cos3[3:4], sin3[3:4])
    q_at_10 = apply_rope(q, cos3[10:11], sin3[10:11])
    k = torch.randn(1, 1, 1, 16)
    k_at_7 = apply_rope(k, cos3[7:8], sin3[7:8])
    k_at_14 = apply_rope(k, cos3[14:15], sin3[14:15])
    dot_early = (q_at_3.flatten() @ k_at_7.flatten()).item()
    dot_late = (q_at_10.flatten() @ k_at_14.flatten()).item()
    assert abs(dot_early - dot_late) < 1e-4
    assert cos.shape == (32, 8)


def _tiny_model_config(**overrides) -> ModelConfig:
    values = dict(
        vocab_size=50,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=192,
        max_sequence_length=64,
        dropout=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_gqa_forward_runs_and_matches_mha_shapes():
    from fontaine.models import build_model

    model = build_model(_tiny_model_config())
    x = torch.randint(0, 50, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 50)


def test_causality_future_tokens_do_not_affect_past():
    from fontaine.models import build_model

    model = build_model(_tiny_model_config())
    model.eval()
    prefix = torch.randint(0, 50, (1, 8))
    future_a = torch.randint(0, 50, (1, 8))
    future_b = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        logits_a = model(torch.cat([prefix, future_a], dim=1)).logits[:, :8]
        logits_b = model(torch.cat([prefix, future_b], dim=1)).logits[:, :8]
    assert torch.allclose(logits_a, logits_b, atol=1e-6)


def test_kv_cache_matches_full_forward():
    from fontaine.models import build_model

    model = build_model(_tiny_model_config())
    model.eval()
    tokens = torch.randint(0, 50, (1, 20))
    with torch.no_grad():
        full = model(tokens).logits
        cache = KVCache.from_config(model.config, batch_size=1, device="cpu")
        parts = []
        for start in range(0, 20, 5):
            chunk = tokens[:, start : start + 5]
            parts.append(model(chunk, cache=cache).logits)
            cache.advance(chunk.shape[1])
        cached = torch.cat(parts, dim=1)
    assert torch.allclose(full, cached, atol=1e-5)


def test_kv_cache_overflow_raises():
    from fontaine.models import build_model

    model = build_model(_tiny_model_config(num_layers=1, max_sequence_length=8))
    model.eval()
    cache = KVCache.from_config(model.config, batch_size=1, device="cpu")
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.randint(0, 50, (1, 9)), cache=cache)


def test_weight_tying():
    from fontaine.models import build_model

    tied = build_model(_tiny_model_config(tie_word_embeddings=True))
    untied = build_model(_tiny_model_config(tie_word_embeddings=False))
    assert tied.lm_head.weight is tied.token_embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.weight
    diff = untied.num_parameters() - tied.num_parameters()
    assert diff == untied.config.vocab_size * untied.config.hidden_size


def test_param_count_matches_analytic_estimate():
    from fontaine.models import build_model
    from fontaine.optimization import estimate_parameter_count

    config = _tiny_model_config(tie_word_embeddings=False)
    model = build_model(config)
    assert model.num_parameters() == estimate_parameter_count(config)["total"]
