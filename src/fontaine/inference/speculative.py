"""Greedy speculative decoding.

The draft model proposes several tokens. The target model keeps the longest
prefix that matches its own argmax, then appends one more target token. With
temperature 0 the accepted text is exactly what the target would have produced
alone. Non-greedy speculative sampling is not implemented.
"""

import torch

from fontaine.generation.sampling import SamplingConfig, sample_token
from fontaine.models.kv_cache import build_kv_cache
from fontaine.models.transformer import FontaineModel


def speculative_ids(
    target: FontaineModel,
    draft: FontaineModel,
    prompt_ids: list[int],
    sampling: SamplingConfig,
    draft_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    eos_id: int,
    generator: torch.Generator | None = None,
) -> list[int]:
    if draft_tokens < 1:
        raise ValueError(f"draft_tokens must be >= 1, got {draft_tokens}")
    if sampling.temperature != 0:
        raise ValueError(
            "speculative decoding is implemented for greedy sampling "
            "(temperature 0). Set temperature to 0, or set speculative_tokens to 0."
        )
    agreed = list(prompt_ids)
    produced: list[int] = []
    while len(produced) < sampling.max_new_tokens:
        draft_logits, draft_cache = _prefill(draft, agreed, device, dtype)
        proposals: list[int] = []
        logits = draft_logits
        room = min(draft_tokens, sampling.max_new_tokens - len(produced))
        for _ in range(room):
            token = sample_token(logits, sampling, produced + proposals, generator)
            if token == eos_id:
                break
            proposals.append(token)
            one = torch.tensor([[token]], dtype=torch.long, device=device)
            logits = draft(one, cache=draft_cache).logits[0, -1].float()
            draft_cache.advance(1)
        target_logits, target_cache = _prefill(target, agreed, device, dtype)
        if not proposals:
            token = sample_token(target_logits, sampling, produced, generator)
            if token == eos_id:
                break
            produced.append(token)
            agreed.append(token)
            continue
        taken: list[int] = []
        logits = target_logits
        rejected = False
        for proposed in proposals:
            choice = sample_token(logits, sampling, produced + taken, generator)
            if choice != proposed:
                taken.append(choice)
                rejected = True
                break
            taken.append(proposed)
            one = torch.tensor([[proposed]], dtype=torch.long, device=device)
            logits = target(one, cache=target_cache).logits[0, -1].float()
            target_cache.advance(1)
        if not rejected and len(produced) + len(taken) < sampling.max_new_tokens:
            bonus = sample_token(logits, sampling, produced + taken, generator)
            if bonus != eos_id:
                taken.append(bonus)
        if not taken:
            break
        produced.extend(taken)
        agreed.extend(taken)
    return produced


def _prefill(
    model: FontaineModel,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, object]:
    cache = build_kv_cache(model.config, 1, device, dtype, "dynamic")
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(tokens, cache=cache).logits[0, -1].float()
    cache.advance(len(token_ids))
    return logits, cache
