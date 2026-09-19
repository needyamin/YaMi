"""Checkpoint format version and integrity errors."""

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreadable, or incompatible."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint file hashes do not match the recorded metadata."""
