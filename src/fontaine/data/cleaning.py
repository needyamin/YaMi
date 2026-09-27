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
    dropped_near_duplicates: int = 0
    dropped_malformed: int = 0
    dropped_script: int = 0
    dropped_contamination: int = 0
    redacted_pii: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_seen": self.seen,
            "documents_kept": self.kept,
            "dropped_empty": self.dropped_empty,
            "dropped_too_short": self.dropped_too_short,
            "dropped_too_long": self.dropped_too_long,
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_near_duplicates": self.dropped_near_duplicates,
            "dropped_malformed": self.dropped_malformed,
            "dropped_script": self.dropped_script,
            "dropped_contamination": self.dropped_contamination,
            "redacted_pii": self.redacted_pii,
        }


@dataclass
class DocumentCleaner:
    """Streams documents through validate → clean → dedup → filter → normalize."""

    config: DataConfig
    stats: CleaningStats = field(default_factory=CleaningStats)

    def clean(self, documents: Iterator[str]) -> Iterator[str]:
        from fontaine.data.quality import (
            ContaminationFilter,
            NearDeduper,
            is_malformed,
            redact_pii,
            script_matches,
        )
        from fontaine.data.sources import iter_source_documents

        seen_hashes: set[bytes] = set()
        min_chars = self.config.min_document_chars
        max_chars = self.config.max_document_chars
        near = NearDeduper(self.config.near_dedup_max_distance) if self.config.near_dedup else None
        contamination = None
        if self.config.contamination_paths:
            contamination = ContaminationFilter(
                iter_source_documents(self.config.contamination_paths),
                self.config.contamination_shingle_fraction,
            )

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
            if self.config.drop_malformed and is_malformed(document):
                self.stats.dropped_malformed += 1
                continue
            if self.config.drop_pii:
                document, hits = redact_pii(document)
                self.stats.redacted_pii += hits
                if len(document) < min_chars:
                    self.stats.dropped_too_short += 1
                    continue
            if self.config.script_filter and not script_matches(document, self.config.script_filter):
                self.stats.dropped_script += 1
                continue
            if near is not None and near.is_duplicate(document):
                self.stats.dropped_near_duplicates += 1
                continue
            if contamination is not None and contamination.is_contaminated(document):
                self.stats.dropped_contamination += 1
                continue

            self.stats.kept += 1
            yield document

        logger.info("cleaning finished: %s", self.stats.as_dict())
