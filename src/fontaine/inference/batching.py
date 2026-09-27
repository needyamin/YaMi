"""A small continuous batch: sequences join and leave between decode steps."""

from dataclasses import dataclass, field

import torch

from fontaine.generation.sampling import SamplingConfig, sample_token
from fontaine.models.kv_cache import build_kv_cache
from fontaine.models.transformer import FontaineModel


@dataclass
class _Request:
    token_ids: list[int]
    max_new: int
    generated: list[int] = field(default_factory=list)
    done: bool = False


class ContinuousBatcher:
    """Decode one token for every active request, and accept new prompts between steps.

    Each request has its own dynamic cache. This is the batching loop, not a
    paged kernel. Finished requests are removed on the next step.
    """

    def __init__(self, model: FontaineModel, eos_id: int, sampling: SamplingConfig) -> None:
        self.model = model
        self.eos_id = eos_id
        self.sampling = sampling
        self._requests: list[_Request] = []
        self._caches: list[object] = []
        self._device = next(model.parameters()).device

    def add(self, token_ids: list[int], max_new: int) -> int:
        request = _Request(token_ids=list(token_ids), max_new=max_new)
        cache = build_kv_cache(self.model.config, 1, self._device, torch.float32, "dynamic")
        self._requests.append(request)
        self._caches.append(cache)
        return len(self._requests) - 1

    def step(self) -> list[tuple[int, list[int]]]:
        """Advance every active request by one token. Returns finished (index, ids)."""
        finished: list[tuple[int, list[int]]] = []
        for index, (request, cache) in enumerate(zip(self._requests, self._caches, strict=True)):
            if request.done:
                continue
            if not request.generated:
                tokens = torch.tensor([request.token_ids], dtype=torch.long, device=self._device)
            else:
                tokens = torch.tensor([[request.generated[-1]]], dtype=torch.long, device=self._device)
            logits = self.model(tokens, cache=cache).logits[0, -1].float()
            cache.advance(tokens.shape[1])
            nxt = sample_token(logits, self.sampling, request.generated, None)
            if nxt == self.eos_id or len(request.generated) + 1 >= request.max_new:
                request.done = True
                if nxt != self.eos_id:
                    request.generated.append(nxt)
                finished.append((index, list(request.generated)))
            else:
                request.generated.append(nxt)
        return finished

    @property
    def active(self) -> int:
        return sum(1 for request in self._requests if not request.done)
