"""Sample documents from a weighted mixture without loading every source at once."""

import random
from collections.abc import Iterator
from pathlib import Path

from fontaine.config.errors import ConfigError
from fontaine.data.sources import iter_source_documents


def iter_mixture(entries: list[dict], seed: int) -> Iterator[str]:
    """Yield documents with probability proportional to ``weight``.

    Each source is streamed independently. A document is kept when a uniform
    draw falls under ``weight / total``. Sources with no files raise.
    """
    if not entries:
        raise ConfigError("data.mixture is empty")
    total = sum(float(entry["weight"]) for entry in entries)
    rng = random.Random(seed)
    streams = []
    for entry in entries:
        path = entry["path"]
        if not Path(path).exists():
            raise ConfigError(
                f"mixture source {entry['name']!r} path does not exist: {path}"
            )
        streams.append((float(entry["weight"]) / total, iter_source_documents([path])))
    for probability, stream in streams:
        for document in stream:
            if rng.random() < probability:
                yield document
