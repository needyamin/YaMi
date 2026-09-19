"""Loss / perplexity evaluator over a prepared token-shard split."""

import math
from typing import Any

import torch

from fontaine.evaluation.base import EvalContext, register_evaluator


@register_evaluator("validation_loss")
class ValidationLossEvaluator:
    """Held-out cross-entropy and perplexity on the evaluation split.

    Metrics are token-weighted (each token contributes equally), which keeps
    perplexity comparable across differently sized evaluation batches.
    """

    name = "validation_loss"

    def __init__(self, context: EvalContext, max_batches: int = 64) -> None:
        self.context = context
        self.max_batches = max_batches
        if context.eval_loader is None:
            raise ValueError("validation_loss evaluator requires an eval_loader in EvalContext")

    def run(self, model: Any, context: EvalContext) -> dict[str, float]:
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(context.eval_loader):
                if batch_index >= self.max_batches:
                    break
                input_ids = batch["input_ids"].to(context.device)
                labels = batch["labels"].to(context.device)
                output = model(input_ids, targets=labels)
                # count non-ignored tokens for correct averaging
                n_tokens = (labels != -100).sum().item()
                if n_tokens == 0:
                    continue
                total_loss += output.loss.item() * n_tokens
                total_tokens += n_tokens
        if was_training:
            model.train()
        if total_tokens == 0:
            raise RuntimeError("validation split yielded no evaluable tokens")
        loss = total_loss / total_tokens
        return {
            "val_loss": loss,
            "val_perplexity": math.exp(min(loss, 20)),  # cap to avoid overflow
            "eval_tokens": float(total_tokens),
        }
