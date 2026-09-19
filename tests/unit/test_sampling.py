"""Sampling strategy tests (pure functions, deterministic)."""

import pytest
import torch

from fontaine.generation import (
    SamplingConfig,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_token,
)


def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 3.0, 0.2, 1.0])
    config = SamplingConfig(temperature=0.0)
    assert sample_token(logits, config, []) == 1


def test_top_k_masks_everything_else():
    logits = torch.tensor([1.0, 5.0, 4.0, 0.1, 2.0])
    masked = apply_top_k(logits, top_k=2)
    assert torch.isinf(masked).sum() == 3
    assert masked[1] == 5.0 and masked[2] == 4.0


def test_top_k_zero_disables():
    logits = torch.randn(100)
    assert torch.equal(apply_top_k(logits, 0), logits)


def test_top_p_keeps_nucleus():
    logits = torch.tensor([10.0, 2.0, 1.0, 0.1])  # softmax ≈ [1.0, ~0, ~0, ~0]
    masked = apply_top_p(logits, top_p=0.9)
    assert not torch.isinf(masked[0])
    assert torch.isinf(masked[1:]).all()


def test_top_p_always_keeps_best_token():
    logits = torch.randn(50)
    masked = apply_top_p(logits, top_p=0.01)
    assert not torch.isinf(masked[logits.argmax()])


def test_repetition_penalty_divides_positive_scores():
    logits = torch.tensor([2.0, 1.0])
    penalized = apply_repetition_penalty(logits, [0], penalty=2.0)
    assert penalized[0] == 1.0  # 2.0 / 2.0
    assert penalized[1] == 1.0


def test_repetition_penalty_boosts_negative_scores():
    logits = torch.tensor([-2.0, 1.0])
    penalized = apply_repetition_penalty(logits, [0], penalty=2.0)
    assert penalized[0] == -4.0


def test_sampling_respects_seed():
    """The engine creates one seeded Generator per generation; sampling from
    two identical generators must produce identical token streams."""
    logits = torch.randn(1000)
    config = SamplingConfig(temperature=1.0, top_k=50, top_p=0.9)
    first, second = [], []
    gen_a = torch.Generator().manual_seed(3)
    gen_b = torch.Generator().manual_seed(3)
    for _ in range(20):
        first.append(sample_token(logits, config, [], gen_a))
        second.append(sample_token(logits, config, [], gen_b))
    assert first == second


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        SamplingConfig(temperature=-1).validate()
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0).validate()
    with pytest.raises(ValueError):
        SamplingConfig(repetition_penalty=0.5).validate()
