"""Small, dependency-light shared utilities."""

from fontaine.utils.git import current_git_commit
from fontaine.utils.hashing import sha256_bytes, sha256_file
from fontaine.utils.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    read_json,
)
from fontaine.utils.logging import configure_logging, get_logger
from fontaine.utils.seeding import (
    collect_rng_states,
    restore_rng_states,
    seed_everything,
)

__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "collect_rng_states",
    "configure_logging",
    "current_git_commit",
    "ensure_dir",
    "get_logger",
    "read_json",
    "restore_rng_states",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
]
