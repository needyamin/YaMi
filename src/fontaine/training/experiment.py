"""Experiment tracking: one self-contained directory per training run.

    experiments/2026-09-19_101530_fontaine-tiny/
    ├── config.yaml        # fully resolved configuration snapshot
    ├── environment.json   # git commit, package versions, device, seed
    ├── metrics.jsonl      # append-only metric log (step-ordered)
    ├── summary.json       # written at the end of the run
    ├── logs/run.log       # full log output
    └── checkpoints/       # CheckpointManager root

Everything needed to reproduce a run is captured at launch; nothing depends
on an external tracking service (though one can be added behind the same
``log_metrics`` seam).
"""

import json
import platform
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fontaine import __version__
from fontaine.config.loader import load_yaml_dict  # noqa: F401  (re-export use)
from fontaine.config.schema import FontaineConfig
from fontaine.utils.git import current_git_commit
from fontaine.utils.io import atomic_write_json, ensure_dir
from fontaine.utils.logging import get_logger

logger = get_logger("training")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-")
    return slug or "run"


class ExperimentRun:
    """Creates and owns the artifact directory of one training run."""

    def __init__(self, directory: Path, config: FontaineConfig) -> None:
        self.directory = directory
        self.config = config
        self.checkpoints_dir = directory / "checkpoints"
        self.logs_dir = directory / "logs"
        self._metrics_file = directory / "metrics.jsonl"
        self._start_time = datetime.now(timezone.utc)
        ensure_dir(self.checkpoints_dir)
        ensure_dir(self.logs_dir)

    @classmethod
    def create(
        cls,
        experiments_dir: str | Path,
        run_name: str,
        config: FontaineConfig,
        resume_dir: str | Path | None = None,
    ) -> "ExperimentRun":
        """Create (or attach to, when resuming) an experiment directory."""
        if resume_dir is not None:
            directory = Path(resume_dir)
            if not (directory / "config.yaml").is_file():
                raise FileNotFoundError(
                    f"cannot resume: {directory} does not look like an experiment directory"
                )
            run = cls(directory, config)
            logger.info("resuming experiment in %s", directory)
            return run

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        directory = Path(experiments_dir) / f"{timestamp}_{_slugify(run_name)}"
        counter = 1
        while directory.exists():
            counter += 1
            directory = Path(experiments_dir) / f"{timestamp}_{_slugify(run_name)}_{counter}"
        run = cls(directory, config)
        run._write_snapshots()
        logger.info("created experiment directory %s", directory)
        return run

    def _write_snapshots(self) -> None:
        from fontaine.utils.io import atomic_write_text

        config_text = _config_to_yaml(self.config)
        atomic_write_text(self.directory / "config.yaml", config_text)
        atomic_write_json(self.directory / "environment.json", self._environment())

    def _environment(self) -> dict[str, Any]:
        return {
            "git_commit": current_git_commit(),
            "fontaine_version": __version__,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "seed": self.config.training.seed,
            "started_utc": self._start_time.isoformat(),
        }

    # -- metric logging --------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append metrics to metrics.jsonl (single JSON line per record)."""
        record = {"step": step, "utc": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self._metrics_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self._metrics_file.is_file():
            return []
        records = []
        for line in self._metrics_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    # -- finalization ----------------------------------------------------------

    def finalize(self, final_metrics: dict[str, float] | None = None) -> Path:
        summary = {
            "run_name": self.config.training.run_name,
            "started_utc": self._start_time.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds(),
            "final_metrics": final_metrics or {},
            "environment": self._environment(),
        }
        path = atomic_write_json(self.directory / "summary.json", summary)
        logger.info("experiment summary written to %s", path)
        return path


def _config_to_yaml(config: FontaineConfig) -> str:
    """Serialize the resolved config to YAML (via JSON to avoid extra deps)."""
    import yaml

    return yaml.safe_dump(
        json.loads(json.dumps(asdict(config), default=str)),
        sort_keys=False,
        default_flow_style=False,
    )
