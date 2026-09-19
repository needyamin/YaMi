"""Byte-level BPE tokenizer backed by the Hugging Face ``tokenizers`` library.

This is the recommended production tokenizer: byte-level means no unknown
tokens for any input (every byte is representable), and BPE gives a good
compute/quality trade-off. The HF dependency is isolated to this module —
the rest of Fontaine only uses the abstract ``Tokenizer`` interface, so a
SentencePiece or Unigram implementation can replace it later.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

from fontaine.tokenizer.base import SpecialTokens, Tokenizer
from fontaine.utils.hashing import sha256_file
from fontaine.utils.io import atomic_write_json, read_json

FORMAT_VERSION = 1
_TOKENIZER_FILE = "tokenizer.json"
_META_FILE = "tokenizer_meta.json"


class ByteLevelBPETokenizer(Tokenizer):
    """Wraps an HF fast tokenizer trained from a streaming corpus iterator."""

    name = "hf_bpe"

    def __init__(self, hf_tokenizer, special_tokens: SpecialTokens | None = None) -> None:
        super().__init__(special_tokens)
        import tokenizers

        if not isinstance(hf_tokenizer, tokenizers.Tokenizer):
            raise TypeError("hf_tokenizer must be a tokenizers.Tokenizer instance")
        self._hf = hf_tokenizer
        self._tokenizer_path: Path | None = None
        self._version_cache: str | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: Iterator[str] | Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        lowercase: bool = False,
        extra_special_tokens: list[str] | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> "ByteLevelBPETokenizer":
        """Train byte-level BPE on a streaming corpus (no full-file loading)."""
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import decoders, models, pre_tokenizers, processors, trainers
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "ByteLevelBPETokenizer requires the 'tokenizers' package: "
                "pip install tokenizers (or pip install fontaine-ai[bpe])"
            ) from exc

        specials = special_tokens or SpecialTokens()
        special_list = specials.as_list() + list(extra_special_tokens or [])
        if lowercase:
            corpus = (doc.lower() for doc in corpus)

        hf = HFTokenizer(models.BPE(unk_token=specials.unk))
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf.decoder = decoders.ByteLevel()
        hf.post_processor = processors.ByteLevel(trim_offsets=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_list,
            show_progress=False,
        )
        hf.train_from_iterator(corpus, trainer)
        return cls(hf, specials)

    # -- Tokenizer contract -------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size(with_added_tokens=True)

    @property
    def version(self) -> str:
        if self._version_cache is None:
            if self._tokenizer_path is None:
                # Not yet persisted: hash the serialized representation.
                self._version_cache = self._version_of(self._hf.to_str().encode("utf-8"))
            else:
                # sha256_file already returns the digest string; just shorten it.
                self._version_cache = sha256_file(self._tokenizer_path)[:12]
        return self._version_cache

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._hf.encode(text, add_special_tokens=add_special_tokens)
        return list(encoding.ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._hf.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        encodings = self._hf.encode_batch(list(texts), add_special_tokens=False)
        return [list(e.ids) for e in encodings]

    def token_to_id(self, token: str) -> int | None:
        return self._hf.token_to_id(token)

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_path = directory / _TOKENIZER_FILE
        self._hf.save(str(tokenizer_path))
        atomic_write_json(
            directory / _META_FILE,
            {
                "format_version": FORMAT_VERSION,
                "type": self.name,
                "special_tokens": {
                    "pad": self.special_tokens.pad,
                    "bos": self.special_tokens.bos,
                    "eos": self.special_tokens.eos,
                    "unk": self.special_tokens.unk,
                },
            },
        )
        self._tokenizer_path = tokenizer_path
        self._version_cache = None
        return tokenizer_path

    @classmethod
    def load(cls, directory: str | Path) -> "ByteLevelBPETokenizer":
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "loading a byte-level BPE tokenizer requires 'pip install tokenizers'"
            ) from exc

        directory = Path(directory)
        meta = read_json(directory / _META_FILE)
        if meta.get("type") != cls.name:
            raise ValueError(f"expected a '{cls.name}' tokenizer, found {meta.get('type')!r}")
        specials = SpecialTokens(**meta["special_tokens"])
        tokenizer = cls(HFTokenizer.from_file(str(directory / _TOKENIZER_FILE)), specials)
        tokenizer._tokenizer_path = directory / _TOKENIZER_FILE
        return tokenizer
