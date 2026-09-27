"""Inference precision: resolution rules, int8 conversion, and generation."""

import pytest
import torch
from torch import nn

from fontaine.config.errors import ConfigError
from fontaine.config.schema import InferenceConfig
from fontaine.inference import Generator
from fontaine.models import KVCache, build_model
from fontaine.optimization import estimate_inference_memory
from fontaine.optimization.quantize import (
    Int8Linear,
    apply_inference_precision,
    estimate_loaded_bytes,
    quantize_int8,
    resolve_inference_precision,
)

CPU = torch.device("cpu")


def test_auto_picks_int8_only_for_larger_cpu_models():
    assert resolve_inference_precision("auto", CPU, 1_000_000) == "fp32"
    assert resolve_inference_precision("auto", CPU, 25_000_000) == "int8"
    assert resolve_inference_precision("fp32", CPU, 25_000_000) == "fp32"
    assert resolve_inference_precision("int8", CPU, 1_000) == "int8"


def test_bf16_on_cpu_falls_back_without_native_support(monkeypatch):
    import fontaine.optimization.quantize as quantize

    monkeypatch.setattr(quantize, "cpu_supports_bf16", lambda: False)
    assert quantize.resolve_inference_precision("bf16", CPU, 1_000) == "fp32"
    monkeypatch.setattr(quantize, "cpu_supports_bf16", lambda: True)
    assert quantize.resolve_inference_precision("bf16", CPU, 1_000) == "bf16"


def test_int8_is_rejected_off_cpu():
    with pytest.raises(ValueError, match="CPU-only"):
        resolve_inference_precision("int8", torch.device("cuda"), 1_000)


def test_inference_config_validates_precision_and_threads():
    InferenceConfig(precision="int8", num_threads=2).validate()
    with pytest.raises(ConfigError, match="precision"):
        InferenceConfig(precision="int4").validate()
    with pytest.raises(ConfigError, match="num_threads"):
        InferenceConfig(num_threads=-1).validate()


@pytest.mark.parametrize("use_torch", [True, False])
def test_int8_model_stays_close_to_fp32(model_config, use_torch):
    torch.manual_seed(0)
    model = build_model(model_config).eval()
    tokens = torch.randint(0, model_config.vocab_size, (1, 12))
    with torch.no_grad():
        reference = model(tokens).logits
    quantized = quantize_int8(model, use_torch=use_torch)
    with torch.no_grad():
        out = quantized(tokens).logits
    assert not any(type(m) is nn.Linear for m in quantized.modules())
    similarity = torch.nn.functional.cosine_similarity(reference.flatten(), out.flatten(), dim=0)
    assert similarity > 0.99


def test_int8_linear_fallback_quantizes_rows():
    linear = nn.Linear(8, 4)
    q = Int8Linear(linear)
    assert q.weight_int8.dtype == torch.int8
    x = torch.randn(3, 8)
    assert torch.allclose(q(x), linear(x), atol=0.05)


def test_bf16_model_runs_with_a_bf16_cache(model_config):
    model = apply_inference_precision(build_model(model_config).eval(), "bf16")
    cache = KVCache.from_config(model_config, 1, CPU, dtype=torch.bfloat16)
    with torch.no_grad():
        logits = model(torch.randint(0, model_config.vocab_size, (1, 6)), cache=cache).logits
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits.float()).all()


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_generator_reports_precision_and_generates(model_config, tokenizer, precision):
    config = InferenceConfig(temperature=0.0, max_new_tokens=10, precision=precision)
    model = build_model(model_config)
    expected_params = model.num_parameters()
    generator = Generator(model, tokenizer, config, device="cpu")
    assert generator.precision == precision
    assert generator.quantization_level == {"fp32": "F32", "int8": "Q8_0"}[precision]
    assert generator.parameter_count == expected_params
    assert isinstance(generator.generate("the fontaine"), str)


def test_int8_size_estimate_matches_the_analytic_estimate(model_config):
    model = build_model(model_config)
    assert estimate_loaded_bytes(model, "fp32") == estimate_inference_memory(model_config)["weights"]
    int8_bytes = estimate_loaded_bytes(model, "int8")
    assert int8_bytes == estimate_inference_memory(model_config, "int8")["weights"]
    assert int8_bytes < estimate_loaded_bytes(model, "fp32")


def test_num_threads_is_applied(model_config, tokenizer):
    before = torch.get_num_threads()
    try:
        generator = Generator(
            build_model(model_config), tokenizer, InferenceConfig(num_threads=1), device="cpu"
        )
        assert generator.num_threads == 1 and torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
