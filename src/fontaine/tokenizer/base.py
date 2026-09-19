"""Abstract tokenizer interface + special-token registry.

The tokenizer is fully independent from the model: a model only ever sees
integer ids and a vocab size. Any class implementing :class:`Tokenizer`
(char-level, byte-level BPE, SentencePiece wrapper, ...) can drive training
and inference without model changes — see ``docs/model/tokenizer.md``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fontaine.utils.hashing import sha256_bytes


@dataclass(frozen=True)
class SpecialTokens:
    """Built-in special tokens, always allocated the first four vocab slots."""

    pad: str = "<|pad|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    unk: str = "<|unk|>"

    def as_list(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]


class Tokenizer(ABC):
    """Base class for all Fontaine tokenizers.

    Contract:
    - ids [0, 1, 2, 3] are always pad/bos/eos/unk (checked on load).
    - ``vocab_size`` includes special tokens; it must match the model config.
    - ``version`` changes whenever the vocabulary or serialization changes,
      and is recorded in datasets, checkpoints, and experiment metadata.
    """

    name: str = "base"

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        """Default batch encoding; subclasses may override for parallelism."""
        return [self.encode(text) for text in texts]

    @abstractmethod
    def save(self, directory: str | Path) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "Tokenizer": ...

    # -- special token ids --------------------------------------------------

    def _special_id(self, token: str) -> int:
        found = self.token_to_id(token)
        if found is None:
            raise ValueError(f"special token {token!r} missing from vocabulary")
        return found

    @property
    def pad_id(self) -> int:
        return self._special_id(self.special_tokens.pad)

    @property
    def bos_id(self) -> int:
        return self._special_id(self.special_tokens.bos)

    @property
    def eos_id(self) -> int:
        return self._special_id(self.special_tokens.eos)

    @property
    def unk_id(self) -> int:
        return self._special_id(self.special_tokens.unk)

    @property
    def all_special_ids(self) -> set[int]:
        return {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

    def token_to_id(self, token: str) -> int | None:
        """Map a literal token string to its id, or None if absent."""
        encoded = self._encode_literal(token)
        return encoded[0] if len(encoded) == 1 else None

    def _encode_literal(self, token: str) -> list[int]:
        # Default: assume single-token specials are produced by plain encode.
        return self.encode(token)

    @staticmethod
    def _version_of(payload: bytes) -> str:
        return sha256_bytes(payload)[:12]
