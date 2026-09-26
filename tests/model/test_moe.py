"""Mixture-of-experts feed-forward: counts, dispatch, and the aux loss."""

import torch
import torch.nn.functional as F

from fontaine.config.schema import ModelConfig
from fontaine.models import build_model
from fontaine.models.components import MixtureOfExperts
from fontaine.optimization import estimate_parameter_count


def _moe_config(**overrides) -> ModelConfig:
    values = dict(
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
    values.update(overrides)
    return ModelConfig(**values)


def test_active_and_total_match_the_instantiated_model():
    config = _moe_config()
    model = build_model(config)
    counts = estimate_parameter_count(config)
    assert model.num_parameters() == counts["total"]
    assert model.num_active_parameters() == counts["active"]
    assert counts["active"] < counts["total"]


def test_top1_runs_each_expert_only_on_its_tokens():
    config = _moe_config(num_layers=1, num_experts=4, num_experts_per_token=1)
    moe = MixtureOfExperts(config)
    tokens = 4
    x = torch.zeros(1, tokens, config.hidden_size)
    for i in range(tokens):
        x[0, i, i] = 1.0
    with torch.no_grad():
        moe.router.weight.zero_()
        for i in range(tokens):
            moe.router.weight[i, i] = 5.0

    seen: dict[int, int] = {}
    for expert_id, expert in enumerate(moe.experts):
        original = expert.forward

        def wrapped(inputs, expert_id=expert_id, original=original):
            seen[expert_id] = inputs.shape[0]
            return original(inputs)

        expert.forward = wrapped  # type: ignore[method-assign]

    y, aux = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(aux)
    assert seen == {0: 1, 1: 1, 2: 1, 3: 1}


def test_aux_loss_is_omitted_when_coef_is_zero():
    config = _moe_config(moe_aux_loss_coef=0.0)
    model = build_model(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, targets=targets)
    manual = F.cross_entropy(out.logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    assert out.loss is not None
    assert torch.allclose(out.loss, manual)


def test_aux_loss_changes_the_training_objective():
    config = _moe_config(moe_aux_loss_coef=1.0)
    model = build_model(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, targets=targets)
    manual = F.cross_entropy(out.logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    assert out.loss is not None and torch.isfinite(out.loss)
    assert not torch.allclose(out.loss, manual)


def test_eval_loss_is_plain_cross_entropy():
    config = _moe_config(moe_aux_loss_coef=1.0)
    model = build_model(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, targets=targets)
    manual = F.cross_entropy(out.logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    assert torch.allclose(out.loss, manual)


def test_gradient_checkpointing_matches_and_router_learns():
    config = _moe_config()
    torch.manual_seed(0)
    plain = build_model(config)
    torch.manual_seed(0)
    checkpointed = build_model(config)
    checkpointed.enable_gradient_checkpointing()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss_plain = plain(input_ids, targets=targets).loss
    loss_ckpt = checkpointed(input_ids, targets=targets).loss
    assert torch.allclose(loss_plain, loss_ckpt)

    loss_plain.backward()
    router_grad = plain.blocks[0].mlp.router.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
