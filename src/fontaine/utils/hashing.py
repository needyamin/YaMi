"""SHA-256 hashing for dataset shards and checkpoint integrity."""

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file (constant memory regardless of file size)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
