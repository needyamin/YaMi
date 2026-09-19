"""Reproducibility helpers: seeding and RNG state capture/restore.

Every training run records the full RNG state in its checkpoints so a resume
continues the same random stream (data order, dropout, sampling).
"""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from fontaine.utils.logging import get_logger

logger = get_logger("utils")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch RNGs; optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic)


@dataclass
class RNGStates:
    """Snapshot of all RNG states relevant to a run.

    The numpy state is stored as plain ints (not ndarrays) so checkpoint
    files stay loadable under ``torch.load(weights_only=True)``.
    """

    python: Any
    numpy: tuple  # ("MT19937", keys: list[int], pos, has_gauss, cached)
    torch: Any
    torch_cuda: Any | None = None


def collect_rng_states() -> RNGStates:
    """Capture current RNG states (called when saving a checkpoint)."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return RNGStates(
        python=random.getstate(),
        numpy=(kind, [int(k) for k in keys], int(pos), int(has_gauss), int(cached)),
        torch=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_states(states: RNGStates) -> None:
    """Restore RNG states (called when resuming from a checkpoint)."""
    random.setstate(states.python)
    kind, keys, pos, has_gauss, cached = states.numpy
    np.random.set_state((kind, np.array(keys, dtype=np.uint32), int(pos), int(has_gauss), int(cached)))
    torch.set_rng_state(states.torch)
    if states.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states.torch_cuda)
