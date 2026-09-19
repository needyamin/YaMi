"""Trainer integration tests: overfitting sanity check and resume."""

import json

import torch

from fontaine.config.schema import DataConfig, FontaineConfig, TokenizerConfig, TrainingConfig
from fontaine.data import TokenShardDataset
from fontaine.models import build_model
from fontaine.training import ExperimentRun, Trainer


def _make_config(model_config, prepared_dir, tmp_path, **training_overrides) -> FontaineConfig:
    defaults = dict(
        run_name="trainer-test",
        max_steps=100,
        batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        warmup_steps=10,
        eval_interval=50,
        log_interval=10,
        checkpoint_interval=50,
        keep_last_n_checkpoints=2,
        seed=7,
    )
    defaults.update(training_overrides)
    training = TrainingConfig(
        experiments_dir=str(tmp_path / "experiments"),
        **defaults,
    )
    return FontaineConfig(
        model=model_config,
        tokenizer=TokenizerConfig(type="char"),
        data=DataConfig(manifest_path=str(prepared_dir / "manifest.json"), sequence_length=32),
        training=training,
    )


def _build_trainer(config: FontaineConfig, tokenizer, prepared_dir, tmp_path) -> Trainer:
    from torch.utils.data import DataLoader

    train_dataset = TokenShardDataset(prepared_dir, "train", 32)
    eval_dataset = TokenShardDataset(prepared_dir, "val", 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.batch_size)
    run = ExperimentRun.create(config.training.experiments_dir, config.training.run_name, config)
    model = build_model(config.model)
    return Trainer(config, model, tokenizer, train_loader, eval_loader, run)


def test_tiny_model_overfits_tiny_dataset(model_config, tokenizer, prepared_dir, tmp_path):
    """The classic sanity check: a small model must be able to overfit."""
    config = _make_config(
        model_config,
        prepared_dir,
        tmp_path,
        max_steps=250,
        learning_rate=5e-3,
        warmup_steps=5,
    )
    trainer = _build_trainer(config, tokenizer, prepared_dir, tmp_path)
    final = trainer.fit()

    records = trainer.run.read_metrics()
    train_losses = [r["train_loss"] for r in records if "train_loss" in r]
    assert len(train_losses) >= 10
    first, last = train_losses[0], train_losses[-1]
    assert last < first * 0.4, f"loss did not fall enough: {first:.3f} -> {last:.3f}"
    assert last < 1.0, f"overfit threshold not reached: {last:.3f}"
    assert "val_loss" in final and "val_perplexity" in final
    assert final["val_perplexity"] < 4.0

    # experiment artifacts exist
    assert (trainer.run.directory / "config.yaml").is_file()
    assert (trainer.run.directory / "environment.json").is_file()
    assert (trainer.run.directory / "metrics.jsonl").is_file()
    assert list((trainer.run.directory / "checkpoints").glob("step_*"))


def test_resume_continues_step_counting(model_config, tokenizer, prepared_dir, tmp_path):
    config = _make_config(model_config, prepared_dir, tmp_path, max_steps=80, checkpoint_interval=80)
    trainer = _build_trainer(config, tokenizer, prepared_dir, tmp_path)
    trainer.fit()

    config2 = _make_config(model_config, prepared_dir, tmp_path, max_steps=160, checkpoint_interval=80)
    trainer2 = _build_trainer(config2, tokenizer, prepared_dir, tmp_path)
    # resume attaches to the *existing* experiment directory
    run_dir = trainer.run.directory
    trainer2.run = ExperimentRun.create(
        config2.training.experiments_dir, config2.training.run_name, config2, resume_dir=run_dir
    )
    trainer2.checkpoints.__init__(run_dir / "checkpoints", keep_last_n=2)  # type: ignore[misc]
    trainer2.fit(resume="latest")

    records = trainer2.run.read_metrics()
    steps = [r["step"] for r in records]
    assert max(steps) == 160
    assert any(s <= 80 for s in steps)  # history preserved


def test_evaluator_registry_extension_point(model_config, tokenizer, prepared_dir, tmp_path):
    """A custom benchmark registers itself and runs without touching the Trainer."""
    from fontaine.evaluation import EvalContext, register_evaluator, registered_evaluators

    @register_evaluator("constant_benchmark")
    def _factory(context: EvalContext, value: float = 1.0):
        class _ConstantBenchmark:
            name = "constant_benchmark"

            def run(self, model, context):
                return {"custom_score": value}

        return _ConstantBenchmark()

    try:
        assert "constant_benchmark" in registered_evaluators()
        config = _make_config(
            model_config, prepared_dir, tmp_path, max_steps=10, eval_interval=10, checkpoint_interval=10
        )
        config.evaluation.evaluators = [{"name": "constant_benchmark", "params": {"value": 3.5}}]
        trainer = _build_trainer(config, tokenizer, prepared_dir, tmp_path)
        metrics = trainer.evaluate()
        assert metrics["custom_score"] == 3.5
    finally:
        from fontaine.evaluation.base import _EVALUATOR_REGISTRY

        _EVALUATOR_REGISTRY.pop("constant_benchmark", None)


def test_metrics_jsonl_is_step_ordered(model_config, tokenizer, prepared_dir, tmp_path):
    config = _make_config(
        model_config, prepared_dir, tmp_path, max_steps=30, log_interval=10, eval_interval=30, checkpoint_interval=30
    )
    trainer = _build_trainer(config, tokenizer, prepared_dir, tmp_path)
    trainer.fit()
    lines = (trainer.run.directory / "metrics.jsonl").read_text().strip().splitlines()
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == sorted(steps)
