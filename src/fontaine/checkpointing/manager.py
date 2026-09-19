"""Checkpoint manager: save / load / resume / best / latest / prune.

Layout (``docs/training/checkpoint-format.md``)::

    <root>/step_000000200/model.pt        # model state dict (fp32 master weights)
    <root>/step_000000200/optimizer.pt    # optimizer + scheduler + scaler state
    <root>/step_000000200/rng.pt          # python/numpy/torch RNG states
    <root>/step_000000200/meta.json       # format version, config snapshot,
                                          # tokenizer/dataset versions, git commit,
                                          # SHA-256 of every file
    <root>/latest.json                    # pointer {"step": N, "path": "step_..."}
    <root>/best.json                      # pointer {"step": N, "path", "metric", "value"}

Design decisions:
- Atomic writes (temp file + rename) so a crash never corrupts a checkpoint.
- Pointer JSON files instead of symlinks: symlinks are unreliable on Windows
  and forbidden in some filesystems; a one-file indirection is portable.
- Every file is hashed into ``meta.json``; loading verifies integrity by
  default so silent corruption is caught at resume time, not mid-run.
- ``world_size`` is recorded now so future sharded (rank-split) checkpoints
  have a version-compatible evolution path (see docs).
"""

import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fontaine.checkpointing.errors import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointIntegrityError,
)
from fontaine.utils.hashing import sha256_file
from fontaine.utils.io import atomic_write_json, read_json
from fontaine.utils.logging import get_logger
from fontaine.utils.seeding import RNGStates, collect_rng_states, restore_rng_states

logger = get_logger("checkpointing")

_LATEST_POINTER = "latest.json"
_BEST_POINTER = "best.json"
_META_FILE = "meta.json"
_MODEL_FILE = "model.pt"
_OPTIMIZER_FILE = "optimizer.pt"
_RNG_FILE = "rng.pt"


def _to_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detach all parameters to CPU so checkpoints are device-independent."""
    return {name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()}


@dataclass
class CheckpointMetadata:
    """Everything needed to identify and resume a checkpoint."""

    format_version: int
    step: int
    epoch: int
    created_utc: str
    world_size: int = 1
    config: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    metric: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # filename -> sha256

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointMetadata":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})


class CheckpointManager:
    """Owns the checkpoint directory of one training run."""

    def __init__(self, root_dir: str | Path, keep_last_n: int = 2) -> None:
        self.root = Path(root_dir)
        self.keep_last_n = keep_last_n
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------

    def _step_dir(self, step: int) -> Path:
        return self.root / f"step_{step:08d}"

    def _resolve(self, source: str | Path) -> Path:
        """Resolve 'latest' / 'best' pointers or a literal step-dir path."""
        if isinstance(source, Path):
            return source
        if source == "latest":
            return self._pointer_path(_LATEST_POINTER)
        if source == "best":
            return self._pointer_path(_BEST_POINTER)
        return Path(source)

    def _pointer_path(self, name: str) -> Path:
        pointer_file = self.root / name
        if not pointer_file.is_file():
            raise CheckpointError(f"no {name} pointer in {self.root}")
        relative = read_json(pointer_file)["path"]
        resolved = self.root / relative
        if not resolved.is_dir():
            raise CheckpointError(f"pointer {name} references missing directory {resolved}")
        return resolved

    # -- save ----------------------------------------------------------------

    def save(
        self,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        config: dict[str, Any] | None = None,
        tokenizer: dict[str, Any] | None = None,
        dataset: dict[str, Any] | None = None,
        code: dict[str, Any] | None = None,
        metric: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a complete, integrity-checked checkpoint; update pointers."""
        step_dir = self._step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        self._save_torch(step_dir / _MODEL_FILE, {"model": _to_cpu_state_dict(model)}, files)
        optimizer_payload: dict[str, Any] = {"optimizer": None, "scheduler": None, "scaler": None}
        if optimizer is not None:
            optimizer_payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            optimizer_payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            optimizer_payload["scaler"] = scaler.state_dict()
        self._save_torch(step_dir / _OPTIMIZER_FILE, optimizer_payload, files)

        rng_states: RNGStates = collect_rng_states()
        self._save_torch(step_dir / _RNG_FILE, asdict(rng_states), files)

        meta = CheckpointMetadata(
            format_version=CHECKPOINT_FORMAT_VERSION,
            step=step,
            epoch=epoch,
            created_utc=datetime.now(timezone.utc).isoformat(),
            world_size=1,  # distributed runs extend this to per-rank shards
            config=config or {},
            tokenizer=tokenizer or {},
            dataset=dataset or {},
            code=code or {},
            metric=metric or {},
            files=files,
        )
        atomic_write_json(step_dir / _META_FILE, asdict(meta))

        atomic_write_json(self.root / _LATEST_POINTER, {"step": step, "path": step_dir.name})
        if metric and "value" in metric:
            self._maybe_update_best(step_dir, step, metric)
        self._prune()
        logger.info("saved checkpoint step=%d at %s", step, step_dir)
        return step_dir

    def _save_torch(self, path: Path, payload: dict[str, Any], files: dict[str, str]) -> None:
        torch.save(payload, path)
        files[path.name] = sha256_file(path)

    def _maybe_update_best(self, step_dir: Path, step: int, metric: dict[str, Any]) -> None:
        """Update the best pointer when the tracked metric improves (lower=better
        for loss-like metrics, configurable via metric['mode'])."""
        name = metric.get("name", "metric")
        mode = metric.get("mode", "min")
        value = float(metric["value"])
        best_file = self.root / _BEST_POINTER
        if best_file.is_file():
            current = read_json(best_file)
            if current.get("metric", {}).get("name") == name:
                best_value = float(current["metric"]["value"])
                if (mode == "min" and value >= best_value) or (
                    mode == "max" and value <= best_value
                ):
                    return
        atomic_write_json(
            best_file,
            {"step": step, "path": step_dir.name, "metric": metric},
        )

    def _prune(self) -> None:
        """Keep the newest ``keep_last_n`` step dirs plus the best checkpoint."""
        best_name = None
        best_file = self.root / _BEST_POINTER
        if best_file.is_file():
            best_name = read_json(best_file).get("path")
        step_dirs = sorted(
            (d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("step_")),
            key=lambda d: d.name,
        )
        excess = step_dirs[: max(0, len(step_dirs) - self.keep_last_n)]
        for stale in excess:
            if stale.name == best_name:
                continue
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("pruned old checkpoint %s", stale)

    # -- load ----------------------------------------------------------------

    def load(
        self,
        source: str | Path = "latest",
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
        verify_integrity: bool = True,
        restore_rng: bool = True,
    ) -> CheckpointMetadata:
        """Load a checkpoint into the given modules; returns its metadata.

        ``source`` may be 'latest', 'best', a step-directory path, or a
        directory name relative to the manager root.
        """
        resolved = self._resolve(source)
        meta_file = resolved / _META_FILE
        if not meta_file.is_file():
            raise CheckpointError(f"not a checkpoint directory (missing meta.json): {resolved}")
        meta = CheckpointMetadata.from_dict(read_json(meta_file))

        if meta.format_version != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"checkpoint format_version {meta.format_version} != supported "
                f"{CHECKPOINT_FORMAT_VERSION} — use fontaine checkpoint convert"
            )
        if verify_integrity:
            self._verify(resolved, meta)
        if restore_rng and (resolved / _RNG_FILE).is_file():
            restore_rng_states(RNGStates(**torch.load(resolved / _RNG_FILE, map_location=map_location)))

        if model is not None:
            payload = torch.load(resolved / _MODEL_FILE, map_location=map_location)
            model.load_state_dict(payload["model"])
        if optimizer is not None or scheduler is not None or scaler is not None:
            payload = torch.load(resolved / _OPTIMIZER_FILE, map_location=map_location)
            if optimizer is not None and payload.get("optimizer") is not None:
                optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            if scaler is not None and payload.get("scaler") is not None:
                scaler.load_state_dict(payload["scaler"])
        logger.info("loaded checkpoint step=%d from %s", meta.step, resolved)
        return meta

    @staticmethod
    def _verify(step_dir: Path, meta: CheckpointMetadata) -> None:
        for filename, expected_hash in meta.files.items():
            path = step_dir / filename
            if not path.is_file():
                raise CheckpointIntegrityError(f"checkpoint file missing: {path}")
            actual = sha256_file(path)
            if actual != expected_hash:
                raise CheckpointIntegrityError(
                    f"integrity check failed for {path}: "
                    f"recorded={expected_hash} actual={actual}"
                )

    # -- inspection ------------------------------------------------------------

    def inspect(self, source: str | Path = "latest") -> dict[str, Any]:
        """Return checkpoint metadata without loading tensors (for the CLI)."""
        resolved = self._resolve(source)
        meta_file = resolved / _META_FILE
        if not meta_file.is_file():
            raise CheckpointError(f"not a checkpoint directory: {resolved}")
        return read_json(meta_file)
