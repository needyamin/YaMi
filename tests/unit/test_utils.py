"""Utility tests: atomic writes, hashing, seeding, experiment runs."""

import json

import pytest
import torch

from fontaine.utils.git import current_git_commit
from fontaine.utils.hashing import sha256_bytes, sha256_file
from fontaine.utils.io import atomic_write_json, atomic_write_text, read_json
from fontaine.utils.seeding import seed_everything


def test_atomic_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    atomic_write_json(path, {"b": 1, "a": [1, 2, None]})
    assert read_json(path) == {"b": 1, "a": [1, 2, None]}
    text = path.read_text()
    assert json.loads(text) == {"b": 1, "a": [1, 2, None]}


def test_atomic_text_roundtrip(tmp_path):
    path = tmp_path / "file.txt"
    atomic_write_text(path, "héllo utf-8")
    assert path.read_text(encoding="utf-8") == "héllo utf-8"


def test_no_temp_files_left_behind(tmp_path):
    atomic_write_json(tmp_path / "x.json", {"ok": True})
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_sha256_file_matches_bytes(tmp_path):
    path = tmp_path / "blob.bin"
    payload = b"fontaine" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_seeded_models_identical(model_config):
    from fontaine.models import build_model

    seed_everything(321)
    a = build_model(model_config)
    seed_everything(321)
    b = build_model(model_config)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_git_commit_returns_string():
    commit = current_git_commit()
    assert isinstance(commit, str) and len(commit) > 3


def test_experiment_run_lifecycle(tmp_path, model_config):
    from fontaine.config.schema import FontaineConfig
    from fontaine.training import ExperimentRun

    config = FontaineConfig()
    run = ExperimentRun.create(tmp_path / "experiments", "my run!", config)
    assert "my-run" in run.directory.name
    assert (run.directory / "config.yaml").is_file()
    assert (run.directory / "environment.json").is_file()
    run.log_metrics({"train_loss": 1.5}, step=10)
    run.log_metrics({"train_loss": 1.2}, step=20)
    assert [r["step"] for r in run.read_metrics()] == [10, 20]
    run.finalize({"train_loss": 1.2})
    summary = read_json(run.directory / "summary.json")
    assert summary["final_metrics"] == {"train_loss": 1.2}
    assert summary["environment"]["seed"] == config.training.seed


def test_experiment_resume_requires_existing_dir(tmp_path):
    from fontaine.config.schema import FontaineConfig
    from fontaine.training import ExperimentRun

    with pytest.raises(FileNotFoundError, match="cannot resume"):
        ExperimentRun.create(tmp_path, "x", FontaineConfig(), resume_dir=tmp_path / "missing")
