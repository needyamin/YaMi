"""--resume must attach to the experiment directory, not its parent."""

import pytest
import torch

from fontaine.cli.main import dataloader_workers, resolve_resume_target
from fontaine.config.errors import ConfigError


def test_resume_checkpoints_dir_is_the_run(tmp_path):
    run = tmp_path / "experiments" / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "latest.json").write_text('{"step": 1, "path": "step_000000001"}', encoding="utf-8")
    (run / "config.yaml").write_text("ok: true\n", encoding="utf-8")

    experiment, source = resolve_resume_target(checkpoints)

    assert experiment == run
    assert source == "latest"
    assert (experiment / "config.yaml").is_file()


def test_resume_best_pointer_uses_the_same_run(tmp_path):
    run = tmp_path / "experiments" / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "best.json").write_text("{}", encoding="utf-8")

    experiment, source = resolve_resume_target(checkpoints)

    assert experiment == run
    assert source == "best"


def test_resume_step_dir_is_the_run(tmp_path):
    run = tmp_path / "experiments" / "run"
    step = run / "checkpoints" / "step_000000010"
    step.mkdir(parents=True)
    (step / "meta.json").write_text("{}", encoding="utf-8")

    experiment, source = resolve_resume_target(step)

    assert experiment == run
    assert source == step


def test_resume_rejects_a_plain_directory(tmp_path):
    with pytest.raises(ConfigError, match="not a checkpoints directory"):
        resolve_resume_target(tmp_path)


def test_dataloader_workers_follow_the_resolved_device():
    assert dataloader_workers(torch.device("cpu")) == 0
    assert dataloader_workers(torch.device("cuda")) == 2
    assert dataloader_workers(torch.device("mps")) == 0
