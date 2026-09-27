"""Fontaine evaluation subsystem."""

from fontaine.evaluation import tasks as _tasks  # noqa: F401  (registers evaluators)
from fontaine.evaluation.base import (
    EvalContext,
    Evaluator,
    GenerativeBenchmark,
    build_evaluator,
    register_evaluator,
    registered_evaluators,
)
from fontaine.evaluation.loss_eval import ValidationLossEvaluator

__all__ = [
    "EvalContext",
    "Evaluator",
    "GenerativeBenchmark",
    "ValidationLossEvaluator",
    "build_evaluator",
    "register_evaluator",
    "registered_evaluators",
]
