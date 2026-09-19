"""Optimizer and learning-rate schedule factories.

The schedule is implemented manually (not torch LambdaLR) so its state is a
single integer step count — trivial to checkpoint and restore exactly.
"""

import math
from typing import Any

import torch

from fontaine.config.schema import TrainingConfig


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """AdamW with weight-decay groups: decay 2D+ matrices, skip 1D params
    (biases, norm weights) and the token embedding."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("token_embedding.weight"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=1e-8,
    )


class WarmupScheduler:
    """Linear warmup → cosine decay to ``min_ratio * base_lr`` (or constant).

    Stepping is explicit: call :meth:`step` once per optimizer update.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_ratio: float = 0.1,
        schedule: str = "cosine",
    ) -> None:
        if schedule not in ("cosine", "constant"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.min_ratio = min_ratio
        self.schedule = schedule
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._apply(0)

    def _lr_for(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return base_lr * (step + 1) / max(self.warmup_steps, 1)
        if self.schedule == "constant":
            return base_lr
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_ratio + (1.0 - self.min_ratio) * cosine)

    def _apply(self, step: int) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = self._lr_for(base_lr, step)

    def step(self) -> None:
        self.step_count += 1
        self._apply(self.step_count)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = state["step_count"]
        self._apply(self.step_count)
