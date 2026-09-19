"""Character-level tokenizer: dependency-free, deterministic, ideal for tests
and debugging the full pipeline on tiny corpora.

Production runs should use the byte-level BPE tokenizer (``hf_bpe``); this
implementation exists so that tokenizer training is never a bottleneck for
pipeline development, and as the simplest reference implementation of the
``Tokenizer`` contract.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from fontaine.tokenizer.base import SpecialTokens, Tokenizer
from fontaine.utils.io import atomic_write_json, read_json

FORMAT_VERSION = 1


class CharTokenizer(Tokenizer):
    """Maps one unicode character to one token. Vocab = specials + corpus chars."""

    name = "char"

    def __init__(self, chars: list[str], special_tokens: SpecialTokens | None = None) -> None:
        super().__init__(special_tokens)
        self._special_list = self.special_tokens.as_list()
        if set(chars) & set(self._special_list):
            raise ValueError("corpus characters must not collide with special token strings")
        # ids: specials occupy 0..3, corpus chars follow in sorted (stable) order.
        self._id_to_char = self._special_list + sorted(chars)
        self._char_to_id = {c: i for i, c in enumerate(self._id_to_char)}
        self._vocab_payload = json.dumps(self._id_to_char, ensure_ascii=False).encode("utf-8")

    # -- construction -------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        lowercase: bool = False,
        special_tokens: SpecialTokens | None = None,
    ) -> "CharTokenizer":
        """Build the vocabulary from a streaming corpus (constant memory)."""
        specials = special_tokens or SpecialTokens()
        chars: set[str] = set()
        for document in corpus:
            if lowercase:
                document = document.lower()
            chars.update(document)
        chars -= set(specials.as_list())
        return cls(sorted(chars), specials)

    # -- Tokenizer contract -------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_char)

    @property
    def version(self) -> str:
        return self._version_of(self._vocab_payload)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        unk = self._char_to_id[self.special_tokens.unk]
        ids = [self._char_to_id.get(ch, unk) for ch in text]
        if add_special_tokens:
            return [self._char_to_id[self.special_tokens.bos], *ids, self._char_to_id[self.special_tokens.eos]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = set(range(len(self._special_list)))
        return "".join(
            self._id_to_char[i] for i in ids if not (skip_special_tokens and i in specials)
        )

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        payload = {
            "format_version": FORMAT_VERSION,
            "type": self.name,
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "unk": self.special_tokens.unk,
            },
            "vocab": self._id_to_char,
        }
        path = atomic_write_json(directory / "tokenizer.json", payload)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "CharTokenizer":
        payload = read_json(Path(directory) / "tokenizer.json")
        if payload.get("type") != cls.name:
            raise ValueError(f"expected a '{cls.name}' tokenizer, found {payload.get('type')!r}")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported tokenizer format_version {payload.get('format_version')!r}"
            )
        specials = SpecialTokens(**payload["special_tokens"])
        vocab = payload["vocab"]
        return cls(vocab[len(specials.as_list()) :], specials)

    def token_to_id(self, token: str) -> int | None:
        return self._char_to_id.get(token)
