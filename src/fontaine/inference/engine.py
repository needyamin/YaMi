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
from fontaine.tokenizer.base import Tokenizer
from fontaine.utils.logging import get_logger

logger = get_logger("inference")


class Generator:
    """Text generation against a trained Fontaine model.

    Requests are serialized with a lock because the model and KV cache are
    shared; concurrent serving should run one Generator per worker process
    (see ``fontaine.inference.server`` and ``docs/deployment/serving.md``).
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
        self.model = model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

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
        **overrides: object,
    ) -> str | Iterator[str]:
        """Generate a completion for ``prompt``.

        ``stream=True`` returns an iterator yielding text chunks; otherwise the
        full text is returned. Sampling knobs can be overridden per call
        (``temperature=0.2``, ``stop_sequences=[...]``, ...).
        """
        sampling = self._sampling(overrides)
        if max_new_tokens is not None:
            sampling = replace(sampling, max_new_tokens=max_new_tokens)
        iterator = self._stream(prompt, sampling)
        if stream:
            return iterator
        return "".join(iterator)

    # -- internals -------------------------------------------------------------

    def _stream(self, prompt: str, sampling: SamplingConfig) -> Iterator[str]:
        with self._lock:
            generator = (
                torch.Generator(device=self.device.type).manual_seed(sampling.seed)
                if sampling.seed is not None
                else None
            )
            prompt_ids = self.tokenizer.encode(prompt)
            headroom = self.model.config.max_sequence_length - sampling.max_new_tokens
            if len(prompt_ids) > max(headroom, 1):
                logger.warning(
                    "prompt (%d tokens) truncated from the left to fit the context window",
                    len(prompt_ids),
                )
                prompt_ids = prompt_ids[-max(headroom, 1) :]

            cache = KVCache.from_config(
                self.model.config, batch_size=1, device=self.device, dtype=torch.float32
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            generated: list[int] = []

            with torch.inference_mode():
                # Prefill: process the whole prompt at once, take last-position logits.
                logits = self.model(input_ids, cache=cache).logits[0, -1]
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
                        logits = self.model(one, cache=cache).logits[0, -1]
                        cache.advance(1)


def _first_stop_cut(text: str, stop_sequences: list[str]) -> str | None:
    """Return text truncated at the earliest stop sequence, if one appears."""
    hits = [pos for pos in (text.find(seq) for seq in stop_sequences if seq) if pos != -1]
    if not hits:
        return None
    return text[: min(hits)]
