"""Filesystem helpers with atomic-write semantics.

Checkpoints, manifests, and experiment metadata must never be observed
half-written, so all JSON/text artifacts are written to a temporary file in
the same directory and then moved with ``os.replace`` (atomic on POSIX and
Windows for same-volume renames).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; returns it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: str | Path, payload: bytes | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    mode = "wb" if isinstance(payload, bytes) else "w"
    kwargs = {} if isinstance(payload, bytes) else {"encoding": "utf-8"}
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode, **kwargs) as handle:  # type: ignore[arg-type]
            handle.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        tmp = Path(tmp_path)
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """Write JSON atomically with stable formatting (sorted keys, 2-space indent)."""
    text = json.dumps(data, indent=2, sort_keys=True, default=str)
    return _atomic_write(path, text)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text atomically."""
    return _atomic_write(path, text)


def read_json(path: str | Path) -> Any:
    import json as _json

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return _json.loads(path.read_text(encoding="utf-8"))
