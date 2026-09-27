"""Training stages other than the pretraining loop.

Supervised fine-tuning masks prompt tokens. Preference training is DPO.
Reinforcement learning is REINFORCE with a verifier reward. Each one runs
the update it names; none of them pretend to be another algorithm.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fontaine.config.errors import ConfigError
from fontaine.config.schema import FontaineConfig
from fontaine.tokenizer.base import Tokenizer


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Direct Preference Optimization loss. All arguments are sequence log-probabilities."""
    if beta <= 0:
        raise ValueError(f"DPO beta must be > 0, got {beta}")
    chosen = beta * (policy_chosen - reference_chosen)
    rejected = beta * (policy_rejected - reference_rejected)
    return -F.logsigmoid(chosen - rejected).mean()


def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE: ``-advantage * log_prob``. Rewards are detached."""
    if log_probs.shape != rewards.shape:
        raise ValueError(
            f"log_probs shape {tuple(log_probs.shape)} != rewards shape {tuple(rewards.shape)}"
        )
    baseline = rewards.mean().detach()
    advantage = rewards.detach() - baseline
    return -(advantage * log_probs).mean()


def sequence_logprob(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Mean token log-probability of ``tokens`` under ``logits`` (next-token aligned)."""
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    gathered = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


def mask_prompt_labels(
    input_ids: torch.Tensor, labels: torch.Tensor, tokenizer: Tokenizer, marker: str = "### Response:\n"
) -> torch.Tensor:
    """Set labels before the response marker to -100.

    ``labels`` are the next-token targets of ``input_ids``. If a row has no
    marker, every label in that row becomes -100 so the prompt is not trained
    as if it were a response.
    """
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    masked = labels.clone()
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].tolist()
        start = _find(ids, marker_ids)
        if start is None:
            masked[row] = -100
            continue
        # labels[i] supervises the token at input position i+1.
        content = start + len(marker_ids)
        cutoff = max(content - 1, 0)
        masked[row, :cutoff] = -100
    return masked


def _find(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    last = len(haystack) - len(needle) + 1
    for index in range(max(last, 0)):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


def run_post_training(config: FontaineConfig, tokenizer_dir: str) -> int:
    """Run DPO or REINFORCE from a JSONL file in ``data.raw_paths``."""
    from fontaine.models import build_model
    from fontaine.tokenizer.registry import load_tokenizer, resolve_vocab_size
    from fontaine.training.optim import build_optimizer

    if not config.data.raw_paths:
        raise ConfigError(
            f"training.stage={config.training.stage} reads JSONL from data.raw_paths. "
            "Point it at a file of records. Pretraining shards are not used."
        )
    path = Path(config.data.raw_paths[0])
    if not path.is_file():
        raise ConfigError(f"post-training data file not found: {path}")
    tokenizer = load_tokenizer(tokenizer_dir)
    config.model.vocab_size = resolve_vocab_size(config.model, tokenizer)
    model = build_model(config.model)
    optimizer = build_optimizer(model, config.training)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ConfigError(f"{path} has no JSONL records")
    if config.training.stage == "preference_training":
        _run_dpo(model, tokenizer, optimizer, records, config.training.max_steps)
    else:
        _run_reinforce(model, tokenizer, optimizer, records, config.training.max_steps)
    return 0


def _encode(tokenizer: Tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor([ids], dtype=torch.long, device=device)


def _run_dpo(model, tokenizer, optimizer, records, steps: int) -> None:
    required = {"prompt", "chosen", "rejected"}
    beta = 0.1
    device = next(model.parameters()).device
    reference = {name: param.detach().clone() for name, param in model.named_parameters()}
    for step in range(steps):
        record = records[step % len(records)]
        missing = required - set(record)
        if missing:
            raise ConfigError(
                f"preference record is missing {sorted(missing)}. "
                "Each line needs prompt, chosen, and rejected."
            )
        chosen = _encode(tokenizer, record["prompt"] + record["chosen"], device)
        rejected = _encode(tokenizer, record["prompt"] + record["rejected"], device)
        policy_chosen = sequence_logprob(model(chosen).logits, chosen)
        policy_rejected = sequence_logprob(model(rejected).logits, rejected)
        with torch.no_grad():
            saved = {name: param.detach().clone() for name, param in model.named_parameters()}
            for name, param in model.named_parameters():
                param.copy_(reference[name])
            ref_chosen = sequence_logprob(model(chosen).logits, chosen)
            ref_rejected = sequence_logprob(model(rejected).logits, rejected)
            for name, param in model.named_parameters():
                param.copy_(saved[name])
        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _run_reinforce(model, tokenizer, optimizer, records, steps: int) -> None:
    from fontaine.evaluation.verifiers import exact_match

    device = next(model.parameters()).device
    model.train()
    for step in range(steps):
        record = records[step % len(records)]
        if "prompt" not in record or "answer" not in record:
            raise ConfigError(
                "reinforcement_learning records need prompt and answer. "
                "The reward is 1 when the sampled completion matches answer, else 0."
            )
        prompt = _encode(tokenizer, record["prompt"], device)
        action_ids = tokenizer.encode(record["answer"], add_special_tokens=False) or [tokenizer.eos_id]
        action = torch.tensor([action_ids], dtype=torch.long, device=device)
        completion = tokenizer.decode(action_ids, skip_special_tokens=True)
        reward = 1.0 if exact_match(completion, record["answer"]) else 0.0
        rl_update(model, optimizer, prompt, action, reward)


def rl_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prompt: torch.Tensor,
    action: torch.Tensor,
    reward: float,
) -> float:
    """One REINFORCE step on ``action`` tokens conditioned on ``prompt``.

    ``action`` is the sampled completion, including no extra batch dimension
    beyond the one already in ``prompt`` (both are ``[1, seq]``).
    """
    tokens = torch.cat([prompt, action], dim=1)
    outputs = model(tokens)
    log_probs = sequence_logprob(outputs.logits, tokens)
    loss = reinforce_loss(log_probs, torch.tensor([reward], device=log_probs.device, dtype=log_probs.dtype))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
