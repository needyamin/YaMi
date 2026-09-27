"""Generative checks that score model text with a verifier.

These are not hidden benchmark suites. Each one states its examples in config
and scores them with exact match or a numeric comparison.
"""

from fontaine.evaluation.base import EvalContext, GenerativeBenchmark, register_evaluator
from fontaine.evaluation.verifiers import exact_match, mathematical


@register_evaluator("exact_match")
class ExactMatchEvaluator(GenerativeBenchmark):
    name = "exact_match"

    def __init__(self, context: EvalContext, items: list[dict] | None = None, max_examples: int = 100) -> None:
        super().__init__(context, max_examples=max_examples)
        self.items = items or []
        if not self.items:
            raise ValueError("exact_match evaluator needs params.items = [{prompt, expected}, ...]")

    def iter_prompts(self) -> list[str]:
        return [str(item["prompt"]) for item in self.items]

    def score(self, prompt: str, completion: str) -> dict[str, float]:
        expected = next(item["expected"] for item in self.items if item["prompt"] == prompt)
        return {"exact_match": 1.0 if exact_match(completion, str(expected)) else 0.0}


@register_evaluator("math")
class MathEvaluator(GenerativeBenchmark):
    name = "math"

    def __init__(self, context: EvalContext, items: list[dict] | None = None, max_examples: int = 100) -> None:
        super().__init__(context, max_examples=max_examples)
        self.items = items or []
        if not self.items:
            raise ValueError("math evaluator needs params.items = [{prompt, expected}, ...]")

    def iter_prompts(self) -> list[str]:
        return [str(item["prompt"]) for item in self.items]

    def score(self, prompt: str, completion: str) -> dict[str, float]:
        expected = next(str(item["expected"]) for item in self.items if item["prompt"] == prompt)
        return {"math_match": 1.0 if mathematical(completion, expected) else 0.0}


def needle_prompt(haystack: str, needle: str, question: str) -> str:
    return f"{haystack}\n{needle}\n{question}"


@register_evaluator("needle")
class NeedleEvaluator(GenerativeBenchmark):
    """Long-context retrieval: the answer is a sentence planted in the prompt."""

    name = "needle"

    def __init__(
        self,
        context: EvalContext,
        haystack: str = "",
        needle: str = "",
        question: str = "",
        expected: str = "",
        max_examples: int = 1,
    ) -> None:
        super().__init__(context, max_examples=max_examples)
        if not (haystack and needle and question and expected):
            raise ValueError("needle evaluator needs haystack, needle, question, and expected")
        self.prompt = needle_prompt(haystack, needle, question)
        self.expected = expected

    def iter_prompts(self) -> list[str]:
        return [self.prompt]

    def score(self, prompt: str, completion: str) -> dict[str, float]:
        return {"needle_match": 1.0 if exact_match(completion, self.expected) else 0.0}
