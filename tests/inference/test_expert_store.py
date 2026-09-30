"""Disk-streamed experts: same outputs, a capped resident set."""

import pytest
import torch

from fontaine.config.errors import ConfigError
from fontaine.config.schema import InferenceConfig, ModelConfig
from fontaine.inference.expert_store import ExpertStore, attach_expert_streaming
from fontaine.models import build_model
from fontaine.models.components import MixtureOfExperts
from fontaine.optimization.expert_tier import (
    tier_decay_value,
    tier_lfru_score,
    tier_pick_lfru,
    tier_should_promote,
)


def _moe_config() -> ModelConfig:
    return ModelConfig(
        architecture="moe_decoder",
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=16,
        num_experts=4,
        num_experts_per_token=1,
        moe_aux_loss_coef=0.01,
        dropout=0.0,
    )


def _expert_state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "gate_proj.weight": torch.randn(8, 4, generator=generator),
        "up_proj.weight": torch.randn(8, 4, generator=generator),
        "down_proj.weight": torch.randn(4, 8, generator=generator),
    }


def _expert_parameter_bytes(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, MixtureOfExperts):
            for expert in module.experts:
                total += sum(parameter.numel() * parameter.element_size() for parameter in expert.parameters())
    return total


def test_promotion_needs_a_margin_and_frequency_outranks_recency():
    assert tier_should_promote(5, 0)
    assert not tier_should_promote(4, 0)
    assert tier_decay_value(7) == 3
    # One frequency step is 256. Recency cannot add more than 255.
    assert tier_lfru_score(1, clock=0, last=0) > tier_lfru_score(0, clock=0, last=0)
    assert tier_lfru_score(2, clock=1000, last=0) > tier_lfru_score(1, clock=1000, last=1000)
    heat = [1, 1, 40]
    last = [0, 0, 0]
    slot, expert_id, gain = tier_pick_lfru(heat, last, clock=0, pinned=[0])
    assert (slot, expert_id) == (0, 2)
    assert gain > 0
    assert tier_pick_lfru([1, 2], [0, 0], clock=0, pinned=[0]) is None


def test_a_leased_expert_stays_resident_when_a_hotter_one_arrives(tmp_path):
    state = _expert_state(0)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    store = ExpertStore(tmp_path, nbytes, [0], overlap=False)
    store.spill(0, 0, state)
    store.spill(0, 1, _expert_state(1))
    held = store.lookup(0, 0)
    for _ in range(30):
        store.release(store.lookup(0, 1))
    again = store.lookup(0, 0)
    assert store.stats().resident_bytes == nbytes
    store.release(held)
    store.release(again)
    store.close()


def test_repeated_use_promotes_an_expert_into_a_full_cache(tmp_path):
    state = _expert_state(0)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    store = ExpertStore(tmp_path, nbytes, [0], overlap=False)
    store.spill(0, 0, state)
    store.spill(0, 1, _expert_state(1))
    for _ in range(8):
        store.release(store.lookup(0, 0))
    for _ in range(20):
        store.release(store.lookup(0, 1))
    before = store.stats().misses
    store.release(store.lookup(0, 1))
    assert store.stats().misses == before
    assert store.stats().resident_bytes == nbytes
    store.close()


def test_streaming_matches_a_resident_forward_and_caps_ram(tmp_path):
    torch.manual_seed(0)
    model = build_model(_moe_config()).eval()
    tokens = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        expected = model(tokens).logits.clone()
    full_bytes = _expert_parameter_bytes(model)
    before = sum(parameter.numel() for parameter in model.parameters())
    store = attach_expert_streaming(model, full_bytes, tmp_path, overlap=False)
    assert store is not None
    after = sum(parameter.numel() for parameter in model.parameters())
    assert after < before
    with torch.no_grad():
        first = model(tokens).logits
        snapshot = store.stats()
        second = model(tokens).logits
        later = store.stats()
    assert torch.allclose(first, expected)
    assert torch.allclose(second, expected)
    assert snapshot.misses > 0
    assert later.misses == snapshot.misses
    assert later.hits > snapshot.hits
    assert 0 < later.resident_bytes <= store.capacity_bytes
    store.close()


def test_a_zero_budget_reloads_every_expert_and_keeps_none(tmp_path):
    torch.manual_seed(1)
    model = build_model(_moe_config()).eval()
    tokens = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        expected = model(tokens).logits.clone()
        store = attach_expert_streaming(model, 0, tmp_path, overlap=False)
        assert store is not None
        output = model(tokens).logits
        model(tokens)
    assert torch.allclose(output, expected)
    stats = store.stats()
    assert stats.resident_bytes == 0
    assert stats.misses > 0
    assert stats.capacity_bytes == 0
    store.close()


def test_a_dense_model_is_left_alone(tmp_path):
    model = build_model(ModelConfig(vocab_size=32, hidden_size=16, num_layers=1, num_attention_heads=4))
    assert attach_expert_streaming(model, 1024, tmp_path, overlap=False) is None


def test_expert_budget_must_be_a_non_negative_integer():
    with pytest.raises(ConfigError, match="expert_budget_mb"):
        InferenceConfig(expert_budget_mb=-1).validate()
    InferenceConfig(expert_budget_mb=0).validate()
    InferenceConfig(expert_budget_mb=512, expert_spill_dir="experts").validate()
