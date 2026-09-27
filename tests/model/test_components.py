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
    counts = estimate_parameter_count(config)
    assert model.num_parameters() == counts["total"]
    assert model.num_active_parameters() == counts["active"]


def test_qk_norm_adds_head_norms_and_matches_estimate():
    from fontaine.models import build_model
    from fontaine.optimization import estimate_parameter_count

    plain = build_model(_tiny_model_config())
    config = _tiny_model_config(qk_norm=True)
    normed = build_model(config)
    assert normed.blocks[0].attn.q_norm is not None
    assert normed.num_parameters() - plain.num_parameters() == 2 * config.head_dim * config.num_layers
    assert normed.num_parameters() == estimate_parameter_count(config)["total"]


def test_layer_window_alternates_local_and_global():
    config = _tiny_model_config(num_layers=4, sliding_window=8, global_attention_every=2)
    assert [config.layer_window(i) for i in range(4)] == [8, 0, 8, 0]
    assert _tiny_model_config(sliding_window=8).layer_window(1) == 8
    assert _tiny_model_config().layer_window(0) == 0


def test_attention_mask_limits_keys_to_the_window():
    from fontaine.models.components import attention_mask

    mask = attention_mask(0, 6, 6, window=3, device=torch.device("cpu"))
    assert mask[5].tolist() == [False, False, False, True, True, True]
    assert mask[1].tolist() == [True, True, False, False, False, False]
    causal = attention_mask(4, 2, 6, window=0, device=torch.device("cpu"))
    assert causal[0].tolist() == [True, True, True, True, True, False]


def test_sliding_window_ignores_tokens_outside_the_window():
    from fontaine.models import build_model

    model = build_model(_tiny_model_config(num_layers=1, sliding_window=4)).eval()
    prefix_a = torch.randint(0, 50, (1, 6))
    prefix_b = torch.randint(0, 50, (1, 6))
    tail = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        out_a = model(torch.cat([prefix_a, tail], dim=1)).logits[:, -1]
        out_b = model(torch.cat([prefix_b, tail], dim=1)).logits[:, -1]
    assert torch.allclose(out_a, out_b, atol=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"qk_norm": True},
        {"sliding_window": 6, "global_attention_every": 2},
        {"rope_scaling_type": "ntk", "rope_scaling_factor": 2.0},
    ],
)
def test_kv_cache_matches_full_forward_with_modern_options(overrides):
    from fontaine.models import build_model

    model = build_model(_tiny_model_config(**overrides)).eval()
    tokens = torch.randint(0, 50, (1, 20))
    with torch.no_grad():
        full = model(tokens).logits
        cache = KVCache.from_config(model.config, batch_size=1, device="cpu")
        parts = [model(tokens[:, :7], cache=cache).logits]
        cache.advance(7)
        for pos in range(7, 20):
            parts.append(model(tokens[:, pos : pos + 1], cache=cache).logits)
            cache.advance(1)
        cached = torch.cat(parts, dim=1)
    assert torch.allclose(full, cached, atol=1e-5)


def test_rope_scaling_linear_and_ntk():
    cos, sin = build_rope_cache(16, 8, theta=10000.0, device="cpu")
    lin_cos, _ = build_rope_cache(
        16, 8, theta=10000.0, device="cpu", scaling_type="linear", scaling_factor=2.0
    )
    assert torch.allclose(lin_cos[4], cos[2], atol=1e-6)
    ntk_cos, _ = build_rope_cache(
        16, 8, theta=10000.0, device="cpu", scaling_type="ntk", scaling_factor=4.0
    )
    assert torch.allclose(ntk_cos[:, 0], cos[:, 0])  # the fastest dimension is unchanged
    assert not torch.allclose(ntk_cos[:, -1], cos[:, -1])
    with pytest.raises(ValueError, match="rope scaling"):
        build_rope_cache(16, 8, theta=10000.0, device="cpu", scaling_type="yarn")


def test_apply_rope_keeps_input_dtype():
    cos, sin = build_rope_cache(4, 8, theta=10000.0, device="cpu")
    x = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16
