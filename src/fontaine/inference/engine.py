"""The inference engine: prompt → tokens → KV-cached decode → text.

Deliberately separate from training: the engine runs in ``eval`` mode with
``inference_mode``, owns a pre-allocated KV cache, and streams text chunks as
they are produced. The evolution path (quantization, continuous batching,
multi-GPU serving) replaces internals, not this interface — see
``docs/inference/architecture.md`` and ``docs/deployment/serving.md``.
"""

import threading
from collections.abc import Iterator
from dataclasses import replace

import torch

from fontaine.config.schema import InferenceConfig
from fontaine.generation import SamplingConfig, sample_token
from fontaine.models import FontaineModel, KVCache
from fontaine.optimization import resolve_device
from fontaine.optimization.quantize import (
    QUANTIZATION_LEVELS,
    apply_inference_precision,
    estimate_loaded_bytes,
    resolve_inference_precision,
    set_num_threads,
)
from fontaine.tokenizer.base import Tokenizer
from fontaine.utils.logging import get_logger

logger = get_logger("inference")


def fit_context(prompt_len: int, max_new_tokens: int, max_sequence_length: int) -> tuple[int, int]:
    """Choose how many prompt tokens to keep and how many tokens to generate.

    If the prompt and the requested answer both fit in the window, keep both.
    Otherwise keep the tail of the prompt (at most 75% of the window) and give
    the remainder to the answer, never more than ``max_new_tokens`` and never
    fewer than one generated token. The returned pair always sums to at most
    ``max_sequence_length``.
    """
    if max_sequence_length < 2:
        raise ValueError(f"max_sequence_length must be >= 2, got {max_sequence_length}")
    if prompt_len < 0 or max_new_tokens < 1:
        raise ValueError("prompt_len must be >= 0 and max_new_tokens must be >= 1")
    if prompt_len + max_new_tokens <= max_sequence_length:
        return prompt_len, max_new_tokens
    prompt_budget = min((max_sequence_length * 3) // 4, max_sequence_length - 1)
    kept = min(prompt_len, max(prompt_budget, 1))
    gen_budget = min(max_new_tokens, max_sequence_length - kept)
    if gen_budget < 1:
        kept = max_sequence_length - 1
        gen_budget = 1
    return kept, gen_budget


class Generator:
    """Text generation against a trained Fontaine model.

    Requests are serialized with a lock because the model and KV cache are
    shared; concurrent serving should run one Generator per worker process
    (see ``fontaine.inference.server`` and ``docs/deployment/serving.md``).

    ``config.precision`` converts the weights once at load time (see
    ``fontaine.optimization.quantize``). Parameter counts and weight size are
    recorded first, because int8 modules no longer expose weights as
    parameters.
    """

    def __init__(
        self,
        model: FontaineModel,
        tokenizer: Tokenizer,
        config: InferenceConfig | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config or InferenceConfig()
        self.tokenizer = tokenizer
        self.device = resolve_device(device)
        self.num_threads = set_num_threads(self.config.num_threads)
        self.parameter_count = model.num_parameters()
        self.active_parameter_count = model.num_active_parameters()
        self.precision = resolve_inference_precision(
            self.config.precision, self.device, self.parameter_count
        )
        self.weight_bytes = estimate_loaded_bytes(model.float(), self.precision)
        model = model.to(self.device)
        model.eval()
        self.model = apply_inference_precision(model, self.precision)
        self.compute_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self._lock = threading.Lock()
        logger.info(
            "inference precision=%s device=%s threads=%d parameters=%d",
            self.precision,
            self.device,
            self.num_threads,
            self.parameter_count,
        )

    @property
    def quantization_level(self) -> str:
        """Ollama-style label for the loaded weight format (F32, BF16, Q8_0)."""
        return QUANTIZATION_LEVELS[self.precision]

    def _sampling(self, overrides: dict[str, object] | None) -> SamplingConfig:
        merged: dict[str, object] = {
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "max_new_tokens": self.config.max_new_tokens,
            "stop_sequences": list(self.config.stop_sequences),
            "seed": self.config.seed,
        }
        merged.update(overrides or {})
        return SamplingConfig(**merged)  # type: ignore[arg-type]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        stream: bool = False,
        *,
        context_length: int | None = None,
        **overrides: object,
    ) -> str | Iterator[str]:
        """Generate a completion for ``prompt``.

        ``stream=True`` returns an iterator yielding text chunks; otherwise the
        full text is returned. Sampling knobs can be overridden per call
        (``temperature=0.2``, ``stop_sequences=[...]``, ...). ``context_length``
        caps the window for this request (the checkpoint maximum still applies).
        """
        sampling = self._sampling(overrides)
        if max_new_tokens is not None:
            sampling = replace(sampling, max_new_tokens=max_new_tokens)
        iterator = self._stream(prompt, sampling, context_length)
        if stream:
            return iterator
        return "".join(iterator)

    def context_budget(
        self,
        prompt: str,
        *,
        context_length: int | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        """Report how a request fits the context window, without generating.

        ``context_length`` is the caller's requested window (Ollama ``num_ctx``).
        It is clamped to the checkpoint maximum. ``truncated`` is true when the
        oldest prompt tokens would be dropped.
        """
        sampling = self._sampling(overrides)
        window = self._resolve_window(context_length)
        with self._lock:
            prompt_len = len(self.tokenizer.encode(prompt))
        kept, gen_budget = fit_context(prompt_len, sampling.max_new_tokens, window)
        return {
            "prompt_tokens": prompt_len,
            "kept_tokens": kept,
            "max_new_tokens": gen_budget,
            "context_length": window,
            "model_context_length": self.model.config.max_sequence_length,
            "truncated": kept < prompt_len,
        }

    # -- internals -------------------------------------------------------------

    def _resolve_window(self, context_length: int | None) -> int:
        """Clamp a requested context window to what this checkpoint can hold."""
        limit = self.model.config.max_sequence_length
        if context_length is None:
            return limit
        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise ValueError("context_length must be an integer >= 2")
        if context_length < 2:
            raise ValueError(f"context_length must be >= 2, got {context_length}")
        if context_length > limit:
            logger.debug("context_length %d clamped to model maximum %d", context_length, limit)
            return limit
        return context_length

    def _stream(
        self, prompt: str, sampling: SamplingConfig, context_length: int | None = None
    ) -> Iterator[str]:
        with self._lock:
            generator = (
                torch.Generator(device=self.device.type).manual_seed(sampling.seed)
                if sampling.seed is not None
                else None
            )
            prompt_ids = self.tokenizer.encode(prompt)
            kept, gen_budget = fit_context(
                len(prompt_ids),
                sampling.max_new_tokens,
                self._resolve_window(context_length),
            )
            if kept < len(prompt_ids):
                logger.warning(
                    "prompt (%d tokens) truncated from the left to %d tokens "
                    "to fit the context window",
                    len(prompt_ids),
                    kept,
                )
                prompt_ids = prompt_ids[-kept:] if kept else []
            if gen_budget < sampling.max_new_tokens:
                logger.warning(
                    "max_new_tokens reduced from %d to %d to fit the context window",
                    sampling.max_new_tokens,
                    gen_budget,
                )
                sampling = replace(sampling, max_new_tokens=gen_budget)

            cache = KVCache.from_config(
                self.model.config, batch_size=1, device=self.device, dtype=self.compute_dtype
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            generated: list[int] = []

            with torch.inference_mode():
                # Prefill: process the whole prompt at once, take last-position logits.
                logits = self.model(input_ids, cache=cache).logits[0, -1].float()
                cache.advance(len(prompt_ids))
                emitted = 0  # chars of the decoded completion already streamed
                for step in range(sampling.max_new_tokens):
                    next_id = sample_token(logits, sampling, generated, generator)
                    if next_id == self.tokenizer.eos_id:
                        return
                    generated.append(next_id)
                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    cut = _first_stop_cut(text, sampling.stop_sequences)
                    if cut is not None:
                        if cut > emitted:
                            yield text[emitted:cut]
                        return
                    # Stream only the newly decoded suffix; byte-level BPE may
                    # briefly produce replacement chars for partial UTF-8.
                    yield text[emitted:]
                    emitted = len(text)
                    if step + 1 < sampling.max_new_tokens:
                        # Decode one token with the KV cache (O(1) per step).
                        one = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                        logits = self.model(one, cache=cache).logits[0, -1].float()
                        cache.advance(1)


def _first_stop_cut(text: str, stop_sequences: list[str]) -> str | None:
    """Return text truncated at the earliest stop sequence, if one appears."""
    hits = [pos for pos in (text.find(seq) for seq in stop_sequences if seq) if pos != -1]
    if not hits:
        return None
    return text[: min(hits)]
