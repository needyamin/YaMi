"""The Fontaine training engine.

The trainer contains zero dataset-specific or model-specific logic. It
composes five collaborators behind narrow interfaces:

- model            forward(input_ids, targets) -> ModelOutput(logits, loss)
- dataloader       yields {"input_ids": LongTensor, "labels": LongTensor}
- optimizer + WarmupScheduler
- CheckpointManager  save/resume with full training state
- evaluators       built from config via the evaluation registry
- ExperimentRun    metrics/config/logs/checkpoints on disk

Adding distributed training (DDP/FSDP/...) means passing a different
``TrainingStrategy`` — the loop below does not change.
"""

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fontaine import __version__
from fontaine.checkpointing import CheckpointManager
from fontaine.config.schema import FontaineConfig
from fontaine.distributed import TrainingStrategy, unwrap_model
from fontaine.evaluation import EvalContext, build_evaluator
from fontaine.models import FontaineModel
from fontaine.optimization import resolve_device, resolve_precision
from fontaine.tokenizer.base import Tokenizer
from fontaine.training.experiment import ExperimentRun
from fontaine.training.optim import WarmupScheduler, build_optimizer
from fontaine.utils.logging import get_logger
from fontaine.utils.seeding import seed_everything

logger = get_logger("training")


class Trainer:
    """Runs the pretraining loop; also the base for future SFT/post-training.

    Extension note: SFT (Phase 7) reuses this class with an instruction-style
    dataset and ``labels`` masking; DPO/RL add new trainer subclasses rather
    than forking this loop (see ``docs/research/post-training.md``).
    """

    def __init__(
        self,
        config: FontaineConfig,
        model: FontaineModel,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader | None,
        run: ExperimentRun,
        strategy: TrainingStrategy | None = None,
    ) -> None:
        self.config = config
        self.train_config = config.training
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.run = run
        self.strategy = strategy or TrainingStrategy()

        self.device = resolve_device(self.train_config.device)
        self.precision = resolve_precision(self.train_config.precision, self.device)
        if self.train_config.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.model.to(self.device)
        self.model = self.strategy.wrap_model(self.model)
        self.raw_model: FontaineModel = unwrap_model(self.model)  # type: ignore[assignment]

        self.optimizer = self.strategy.wrap_optimizer(build_optimizer(self.raw_model, self.train_config))
        self.scheduler = WarmupScheduler(
            self.optimizer,
            total_steps=self.train_config.max_steps,
            warmup_steps=self.train_config.warmup_steps,
            min_ratio=self.train_config.min_learning_rate_ratio,
            schedule=self.train_config.lr_scheduler,
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.precision.use_scaler)
            if self.precision.use_scaler
            else None
        )
        self.checkpoints = CheckpointManager(
            run.checkpoints_dir, keep_last_n=self.train_config.keep_last_n_checkpoints
        )
        self._eval_context = EvalContext(
            device=str(self.device),
            tokenizer=tokenizer,
            eval_loader=eval_loader,
            inference_config=config.inference,
        )
        self.evaluators = [
            build_evaluator(entry, self._eval_context)
            for entry in config.evaluation.evaluators
        ]
        self._last_eval_metrics: dict[str, float] = {}
        self._best_metric: float | None = None
        self._evals_without_improvement = 0

    # -- main loop -------------------------------------------------------------

    def fit(self, resume: str | Path | None = None) -> dict[str, float]:
        """Train from scratch or resume; returns final evaluation metrics."""
        seed_everything(self.train_config.seed, self.train_config.deterministic)
        start_step = 0
        if resume is not None:
            meta = self.checkpoints.load(
                resume,
                model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            start_step = meta.step
            logger.info("resumed from step %d", start_step)

        self.model.train()
        batches: Iterator[dict[str, torch.Tensor]] = self._cycle(self.train_loader)
        effective_batch = self.train_config.batch_size * self.train_config.gradient_accumulation_steps
        tokens_per_step = effective_batch * self.config.data.sequence_length

        final_metrics: dict[str, float] = {}
        for step in range(start_step + 1, self.train_config.max_steps + 1):
            step_loss = self._train_step(batches)
            self.scheduler.step()

            if step % self.train_config.log_interval == 0:
                metrics = {
                    "train_loss": round(step_loss, 6),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "tokens_per_step": tokens_per_step,
                }
                if self.strategy.context.is_primary:
                    logger.info(
                        "step %d/%d loss=%.4f lr=%.2e",
                        step,
                        self.train_config.max_steps,
                        step_loss,
                        self.scheduler.get_last_lr()[0],
                    )
                    self.run.log_metrics(metrics, step)

            if step % self.train_config.eval_interval == 0 or step == self.train_config.max_steps:
                eval_metrics = self.evaluate()
                final_metrics = eval_metrics
                if self.strategy.context.is_primary:
                    logger.info("step %d eval: %s", step, eval_metrics)
                    self.run.log_metrics(eval_metrics, step)
                if self._check_early_stopping(eval_metrics):
                    logger.info("early stopping at step %d", step)
                    break

            if step % self.train_config.checkpoint_interval == 0:
                self.save_checkpoint(step, metric=self._as_checkpoint_metric(final_metrics))

        self.save_checkpoint(
            self.train_config.max_steps,
            metric=self._as_checkpoint_metric(final_metrics),
            final=True,
        )
        if self.strategy.context.is_primary:
            self.run.finalize(final_metrics)
        self.strategy.wait_for_everyone()
        return final_metrics

    # -- internals ---------------------------------------------------------------

    def _cycle(self, loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
        """Infinite batch stream; the seeded DataLoader generator reshuffles
        each epoch deterministically."""
        while True:
            yield from loader

    def _train_step(self, batches: Iterator[dict[str, torch.Tensor]]) -> float:
        """One optimizer step = ``gradient_accumulation_steps`` micro-batches."""
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accum = self.train_config.gradient_accumulation_steps
        autocast_dtype = self.precision.autocast_dtype

        for _ in range(accum):
            batch = next(batches)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                output = self.model(input_ids, targets=labels)
                loss = output.loss / accum
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += output.loss.item()

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.train_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(), self.train_config.max_grad_norm
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return total_loss / accum

    def evaluate(self) -> dict[str, float]:
        """Run all configured evaluators; returns their merged metrics."""
        merged: dict[str, float] = {}
        for evaluator in self.evaluators:
            metrics = evaluator.run(self.raw_model, self._eval_context)
            for key, value in metrics.items():
                if key in merged:
                    raise ValueError(
                        f"evaluator '{evaluator.name}' produced duplicate metric '{key}'"
                    )
                merged[key] = value
        return self.strategy.sync_metrics(merged)

    def _check_early_stopping(self, eval_metrics: dict[str, float]) -> bool:
        patience = self.train_config.early_stopping_patience
        if patience <= 0:
            return False
        tracked = eval_metrics.get("val_loss")
        if tracked is None:
            return False
        if self._best_metric is None or tracked < self._best_metric:
            self._best_metric = tracked
            self._evals_without_improvement = 0
        else:
            self._evals_without_improvement += 1
        return self._evals_without_improvement >= patience

    def _as_checkpoint_metric(self, eval_metrics: dict[str, float]) -> dict[str, Any] | None:
        value = eval_metrics.get("val_loss")
        if value is None:
            return None
        return {"name": "val_loss", "value": value, "mode": "min"}

    def save_checkpoint(
        self,
        step: int,
        metric: dict[str, Any] | None = None,
        final: bool = False,
    ) -> None:
        """Persist full state (primary rank only in distributed runs)."""
        if not self.strategy.context.is_primary:
            return
        dataset_meta: dict[str, Any] = {"manifest_path": self.config.data.manifest_path}
        if self.config.data.manifest_path:
            from fontaine.data.manifest import DatasetManifest

            try:
                manifest = DatasetManifest.load(Path(self.config.data.manifest_path).parent)
                dataset_meta.update(
                    name=manifest.name,
                    version=manifest.version,
                    tokenizer_version=manifest.tokenizer.get("version"),
                    preprocessing_version=manifest.preprocessing_version,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("could not attach dataset metadata to checkpoint: %s", exc)

        self.checkpoints.save(
            step=step,
            epoch=0,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=asdict(self.config),
            tokenizer={
                "name": self.tokenizer.name,
                "version": self.tokenizer.version,
                "vocab_size": self.tokenizer.vocab_size,
            },
            dataset=dataset_meta,
            code={"fontaine_version": __version__},
            metric=metric if not final else self._force_best_metric(metric),
        )

    def _force_best_metric(self, metric: dict[str, Any] | None) -> dict[str, Any] | None:
        """Mark the final checkpoint as best so it survives pruning."""
        if metric is not None and self._best_metric is not None and metric["value"] <= self._best_metric:
            self._best_metric = metric["value"]
        return metric
