"""Model-level forward/backward tests."""

import pytest
import torch

from fontaine.models import build_model


def test_forward_shapes_and_loss(model_config):
    model = build_model(model_config)
    batch, seq = 3, 32
    input_ids = torch.randint(0, model_config.vocab_size, (batch, seq))
    targets = torch.randint(0, model_config.vocab_size, (batch, seq))
    out = model(input_ids, targets=targets)
    assert out.logits.shape == (batch, seq, model_config.vocab_size)
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_loss_ignores_masked_positions(model_config):
    model = build_model(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (1, 16))
    targets = torch.full((1, 16), -100)
    targets[0, 5:] = 3  # only 11 tokens contribute
    out = model(input_ids, targets=targets)
    manual = torch.nn.functional.cross_entropy(
        out.logits[0, 5:].reshape(-1, model_config.vocab_size),
        targets[0, 5:],
        ignore_index=-100,
    )
    assert torch.allclose(out.loss, manual)


def test_gradients_flow_to_all_parameters(model_config):
    model = build_model(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (2, 16))
    loss = model(input_ids, targets=torch.randint(0, model_config.vocab_size, (2, 16))).loss
    loss.backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing == []


def test_gradient_checkpointing_same_loss(model_config):
    torch.manual_seed(0)
    model_a = build_model(model_config)
    torch.manual_seed(0)
    model_b = build_model(model_config)
    model_b.enable_gradient_checkpointing()
    batch = {
        "input_ids": torch.randint(0, model_config.vocab_size, (2, 16)),
    }
    targets = torch.randint(0, model_config.vocab_size, (2, 16))
    loss_a = model_a(batch["input_ids"], targets=targets).loss
    loss_b = model_b(batch["input_ids"], targets=targets).loss
    assert torch.allclose(loss_a, loss_b)


def test_gradient_checkpointing_rejects_cache(model_config):
    from fontaine.models import KVCache

    model = build_model(model_config)
    model.enable_gradient_checkpointing()
    model.train()
    cache = KVCache.from_config(model.config, batch_size=1, device="cpu")
    with pytest.raises(ValueError, match="checkpointing"):
        model(torch.randint(0, model_config.vocab_size, (1, 4)), cache=cache)


def test_seeded_init_is_reproducible(model_config):
    torch.manual_seed(123)
    model_a = build_model(model_config)
    torch.manual_seed(123)
    model_b = build_model(model_config)
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)
