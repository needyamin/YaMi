"""Document filters that compute a real decision.

Near-duplicate detection is 64-bit simhash with banding. The script filter is a
character-set heuristic, not a language-identification model. There is no
learned quality score.
"""

import hashlib
import re
from collections.abc import Iterator

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SCRIPTS = {
    "latin": r"[A-Za-z]",
    "cyrillic": r"[\u0400-\u04FF]",
    "cjk": r"[\u4E00-\u9FFF]",
    "arabic": r"[\u0600-\u06FF]",
    "devanagari": r"[\u0900-\u097F]",
}


def simhash(text: str) -> int:
    """64-bit simhash over word trigrams."""
    words = text.lower().split()
    if len(words) < 3:
        grams = [" ".join(words)] if words else [text]
    else:
        grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    totals = [0] * 64
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        for bit in range(64):
            totals[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, score in enumerate(totals):
        if score > 0:
            result |= 1 << bit
    return result


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class NearDeduper:
    """Drop a document when a previous simhash is within ``max_distance``.

    Banding (4 bands of 16 bits) limits comparisons to documents that share a
    band. This is exact only inside a band; documents that are close but share
    no band are kept.
    """

    def __init__(self, max_distance: int) -> None:
        self.max_distance = max_distance
        self.bands: list[dict[int, list[int]]] = [{} for _ in range(4)]
        self._kept: list[int] = []

    def is_duplicate(self, text: str) -> bool:
        fingerprint = simhash(text)
        candidates: set[int] = set()
        for index in range(4):
            band = (fingerprint >> (index * 16)) & 0xFFFF
            candidates.update(self.bands[index].get(band, []))
        for previous in candidates:
            if hamming(fingerprint, previous) <= self.max_distance:
                return True
        self._kept.append(fingerprint)
        for index in range(4):
            band = (fingerprint >> (index * 16)) & 0xFFFF
            self.bands[index].setdefault(band, []).append(fingerprint)
        return False


def redact_pii(text: str) -> tuple[str, int]:
    """Replace emails, SSNs, and long card-like numbers. Returns text and hit count."""
    hits = 0
    for pattern, token in ((_EMAIL, "[email]"), (_SSN, "[ssn]"), (_CARD, "[number]")):
        text, count = pattern.subn(token, text)
        hits += count
    return text, hits


def is_malformed(text: str) -> bool:
    """Null bytes, or a document that is mostly one repeated character."""
    if "\x00" in text:
        return True
    if len(text) >= 20:
        most = max(text.count(char) for char in set(text))
        if most / len(text) > 0.5:
            return True
    return False


def script_matches(text: str, script: str) -> bool:
    """True when the requested script is a majority of the letters.

    This is a Unicode-range heuristic. It is not a language detector.
    """
    if script not in _SCRIPTS:
        raise ValueError(f"unknown script filter {script!r}")
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    matched = len(re.findall(_SCRIPTS[script], "".join(letters)))
    return matched / len(letters) >= 0.5


def shingles(text: str, size: int = 5) -> set[str]:
    words = text.lower().split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


class ContaminationFilter:
    """Drop a document whose shingles mostly appear in a held-out set."""

    def __init__(self, documents: Iterator[str], fraction: float) -> None:
        self.fraction = fraction
        self._shingles: set[str] = set()
        self._exact: set[bytes] = set()
        for document in documents:
            normalized = " ".join(document.split())
            self._exact.add(hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest())
            self._shingles.update(shingles(normalized))

    def is_contaminated(self, text: str) -> bool:
        normalized = " ".join(text.split())
        exact = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()
        if exact in self._exact:
            return True
        grams = shingles(normalized)
        if not grams or not self._shingles:
            return False
        overlap = sum(1 for gram in grams if gram in self._shingles)
        return overlap / len(grams) >= self.fraction
