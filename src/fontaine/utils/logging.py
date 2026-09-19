"""Logging helpers.

Fontaine uses the standard library ``logging`` module exclusively; heavy
tracking systems (TensorBoard, W&B) plug in through the training layer, not
through global loggers.
"""

import logging
import sys
from pathlib import Path

from fontaine.utils.io import ensure_dir

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger once (console always, optional rotating file)."""
    global _configured
    if _configured:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring defaults on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(f"fontaine.{name}")
