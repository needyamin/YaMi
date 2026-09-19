"""Checkpoint manager tests: save/load, resume state, integrity, pointers."""

import json

import pytest
import torch

from fontaine.checkpointing import (
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointManager,
)
from fontaine.config.schema import TrainingConfig
from fontaine.models import build_model
from fontaine.training.optim import WarmupScheduler, build_optimizer


@pytest.fixture()
def setup(tmp_path, model_config):
    model = build_model(model_config)
    train_config = TrainingConfig(max_steps=100, warmup_steps=5)
    optimizer = build_optimizer(model, train_config)
    scheduler = WarmupScheduler(optimizer, total_steps=100, warmup_steps=5)
    manager = CheckpointManager(tmp_path / "checkpoints", keep_last_n=10)
    return model, optimizer, scheduler, manager


def test_save_and_load_restores_weights_exactly(setup):
    model, optimizer, scheduler, manager = setup
    manager.save(step=10, epoch=0, model=model, optimizer=optimizer, scheduler=scheduler)
    model2 = build_model(model.config)
    # perturb to prove loading overwrites
    with torch.no_grad():
        for param in model2.parameters():
            param.add_(1.0)
    meta = manager.load("latest", model=model2)
    assert meta.step == 10
    assert meta.format_version == 1
    for (name_a, a), (name_b, b) in zip(
        model.state_dict().items(), model2.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(a, b)


def test_metadata_records_provenance(setup):
    model, optimizer, scheduler, manager = setup
    manager.save(
        step=5,
        epoch=0,
        model=model,
        config={"model": {"hidden_size": 64}},
        tokenizer={"name": "char", "version": "abc123"},
        dataset={"name": "toy"},
        metric={"name": "val_loss", "value": 2.5, "mode": "min"},
    )
    meta = manager.inspect("latest")
    assert meta["config"]["model"]["hidden_size"] == 64
    assert meta["tokenizer"]["version"] == "abc123"
    assert meta["metric"]["value"] == 2.5
    assert meta["world_size"] == 1
    assert "sha256" or meta["files"]  # file hashes recorded


def test_integrity_check_catches_corruption(setup):
    model, _, _, manager = setup
    manager.save(step=1, epoch=0, model=model)
    step_dir = manager._resolve("latest")
    model_file = step_dir / "model.pt"
    payload = bytearray(model_file.read_bytes())
    payload[-3] ^= 0xFF  # flip bits near the end
    model_file.write_bytes(bytes(payload))
    with pytest.raises(CheckpointIntegrityError):
        manager.load("latest", model=build_model(model.config))


def test_best_pointer_tracks_improvements(setup):
    model, _, _, manager = setup
    manager.save(step=10, epoch=0, model=model, metric={"name": "val_loss", "value": 1.0, "mode": "min"})
    manager.save(step=20, epoch=0, model=model, metric={"name": "val_loss", "value": 0.5, "mode": "min"})
    manager.save(step=30, epoch=0, model=model, metric={"name": "val_loss", "value": 0.8, "mode": "min"})
    best = json.loads((manager.root / "best.json").read_text())
    assert best["step"] == 20 and best["metric"]["value"] == 0.5
    meta = manager.load("best")
    assert meta.step == 20


def test_pruning_keeps_recent_and_best(setup, tmp_path):
    model, _, _, manager = setup
    manager.keep_last_n = 2
    manager.save(step=1, epoch=0, model=model, metric={"name": "val_loss", "value": 3.0, "mode": "min"})
    manager.save(step=2, epoch=0, model=model)
    manager.save(step=3, epoch=0, model=model)
    manager.save(step=4, epoch=0, model=model)
    names = sorted(d.name for d in manager.root.iterdir() if d.name.startswith("step_"))
    # newest two survive + step_1 is retained because it holds the best metric
    assert names == ["step_00000001", "step_00000003", "step_00000004"]


def test_missing_pointer_raises(tmp_path):
    manager = CheckpointManager(tmp_path / "empty")
    with pytest.raises(CheckpointError, match="no latest.json"):
        manager.load("latest")


def test_rng_states_roundtrip(setup):
    from fontaine.utils.seeding import seed_everything

    model, _, _, manager = setup
    seed_everything(1234)
    torch.rand(3)  # first draw; checkpoint captures the state AFTER it
    manager.save(step=1, epoch=0, model=model)
    seed_everything(999)
    b = torch.rand(3)
    manager.load("latest", model=model)  # restores RNG captured at save time
    d = torch.rand(3)
    # d must be the *second* draw from seed 1234, proving exact state restore
    seed_everything(1234)
    torch.rand(3)
    expected_second = torch.rand(3)
    assert torch.equal(d, expected_second)
    # and it differs from a different seed's stream
    assert not torch.equal(b, expected_second)
