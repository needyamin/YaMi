"""Evaluation framework.

A new benchmark is added by registering a class (or factory) under a name and
listing that name in ``evaluation.evaluators`` — the training engine and CLI
never change. Two families exist:

- Loss-based (run against a token shard split): validation loss/perplexity.
- Generative benchmarks (future: instruction following, QA, summarization):
  subclass :class:`GenerativeBenchmark`, implement ``iter_prompts`` and
  ``score``, and register — the harness handles generation via the standard
  inference stack.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fontaine.utils.logging import get_logger

if TYPE_CHECKING:
    from fontaine.config.schema import InferenceConfig
    from fontaine.inference.engine import Generator
    from fontaine.tokenizer.base import Tokenizer

logger = get_logger("evaluation")


@dataclass
class EvalContext:
    """Everything an evaluator may need, provided by the caller (trainer/CLI)."""

    device: str
    tokenizer: "Tokenizer"
    eval_loader: Any | None = None  # DataLoader over the evaluation split
    generator: "Generator | None" = None  # lazily built by generative benchmarks
    inference_config: "InferenceConfig | None" = None
    max_batches: int = 64


class Evaluator(Protocol):
    """Anything with a stable ``name`` and a ``run`` returning metrics."""

    name: str

    def run(self, model: Any, context: EvalContext) -> dict[str, float]: ...


_EVALUATOR_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_evaluator(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Class/factory decorator: ``@register_evaluator("my_benchmark")``."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _EVALUATOR_REGISTRY:
            raise ValueError(f"evaluator '{name}' is already registered")
        _EVALUATOR_REGISTRY[name] = factory
        return factory

    return decorator


def registered_evaluators() -> list[str]:
    return sorted(_EVALUATOR_REGISTRY)


def build_evaluator(entry: dict[str, Any], context: EvalContext) -> Evaluator:
    """Instantiate one evaluator from a config entry ``{name, params}``."""
    name = entry.get("name")
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(
            f"unknown evaluator '{name}' (registered: {registered_evaluators()}). "
            f"Register new benchmarks in fontaine.evaluation.registry."
        )
    params = dict(entry.get("params", {}))
    return _EVALUATOR_REGISTRY[name](context=context, **params)


class GenerativeBenchmark:
    """Base class for future generation-based benchmarks.

    Subclasses provide prompts and scoring; the harness provides generation
    through the standard inference stack (KV cache, sampling config), so
    benchmarks automatically benefit from inference improvements.
    """

    name: str = "generative_benchmark"

    def __init__(self, context: EvalContext, max_examples: int = 100) -> None:
        self.context = context
        self.max_examples = max_examples
        if context.generator is None:
            raise ValueError(
                f"{self.name} requires a Generator in EvalContext "
                f"(load a checkpoint with fontaine evaluate)"
            )

    def iter_prompts(self) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, prompt: str, completion: str) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def run(self, model: Any, context: EvalContext) -> dict[str, float]:
        assert context.generator is not None
        scores: list[dict[str, float]] = []
        for i, prompt in enumerate(self.iter_prompts()):
            if i >= self.max_examples:
                break
            completion = context.generator.generate(prompt)
            scores.append(self.score(prompt, completion))
        if not scores:
            raise RuntimeError(f"{self.name}: no examples scored")
        averaged = {
            key: sum(s[key] for s in scores) / len(scores) for key in scores[0]
        }
        logger.info("%s: %s", self.name, averaged)
        return averaged
