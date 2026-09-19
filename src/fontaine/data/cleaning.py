"""Streaming document cleaning: validation, normalization, dedup, filtering.

The cleaner is a pure iterator transform: memory use is bounded by the
duplicate-hash set (16 bytes per unique document), not by corpus size. For
web-scale corpora the exact-dedup set is replaced by external tools
(Bloom filters, MinHash LSH) — the pipeline interface stays the same.
"""

import hashlib
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field

from fontaine.config.schema import DataConfig
from fontaine.utils.logging import get_logger

logger = get_logger("data")


@dataclass
class CleaningStats:
    """Counters describing what the cleaner dropped and kept."""

    seen: int = 0
    kept: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_duplicates: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_seen": self.seen,
            "documents_kept": self.kept,
            "dropped_empty": self.dropped_empty,
            "dropped_too_short": self.dropped_too_short,
            "dropped_too_long": self.dropped_too_long,
            "dropped_duplicates": self.dropped_duplicates,
        }


@dataclass
class DocumentCleaner:
    """Streams documents through validate → clean → dedup → filter → normalize."""

    config: DataConfig
    stats: CleaningStats = field(default_factory=CleaningStats)

    def clean(self, documents: Iterator[str]) -> Iterator[str]:
        seen_hashes: set[bytes] = set()
        min_chars = self.config.min_document_chars
        max_chars = self.config.max_document_chars

        for document in documents:
            self.stats.seen += 1
            if self.config.normalize_unicode:
                document = unicodedata.normalize("NFKC", document)
            if self.config.strip_whitespace:
                document = document.strip()

            if not document:
                self.stats.dropped_empty += 1
                continue
            if len(document) < min_chars:
                self.stats.dropped_too_short += 1
                continue
            if len(document) > max_chars:
                self.stats.dropped_too_long += 1
                continue
            if self.config.drop_duplicates:
                fingerprint = hashlib.blake2b(document.encode("utf-8"), digest_size=16).digest()
                if fingerprint in seen_hashes:
                    self.stats.dropped_duplicates += 1
                    continue
                seen_hashes.add(fingerprint)

            self.stats.kept += 1
            yield document

        logger.info("cleaning finished: %s", self.stats.as_dict())
