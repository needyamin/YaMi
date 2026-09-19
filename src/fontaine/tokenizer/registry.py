"""Tokenizer construction from configuration.

Two entry points, mirroring the CLI:
- ``train_tokenizer(config, corpus)`` — build a new tokenizer from raw text.
- ``load_tokenizer(config_or_dir)`` — load a previously trained one.
"""

from collections.abc import Iterable
from pathlib import Path

from fontaine.config.schema import TokenizerConfig
from fontaine.tokenizer.base import Tokenizer
from fontaine.tokenizer.char_level import CharTokenizer
from fontaine.tokenizer.hf_bpe import ByteLevelBPETokenizer


def train_tokenizer(
    config: TokenizerConfig, corpus: Iterable[str], output_dir: str | Path
) -> Tokenizer:
    """Train the configured tokenizer type on a streaming corpus and save it."""
    if config.type == "char":
        tokenizer: Tokenizer = CharTokenizer.train(corpus, lowercase=config.lowercase)
    elif config.type == "hf_bpe":
        tokenizer = ByteLevelBPETokenizer.train(
            iter(corpus),
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            lowercase=config.lowercase,
            extra_special_tokens=config.special_tokens,
        )
    else:
        raise ValueError(
            f"unknown tokenizer type {config.type!r} (expected 'char' or 'hf_bpe')"
        )
    tokenizer.save(Path(output_dir))
    return tokenizer


def load_tokenizer(path: str | Path, tokenizer_type: str | None = None) -> Tokenizer:
    """Load a tokenizer saved under ``path``.

    The stored metadata decides the class; ``tokenizer_type`` optionally
    verifies the expectation (e.g. against ``TokenizerConfig.type``).
    """
    path = Path(path)
    from fontaine.utils.io import read_json

    meta_files = [p for p in path.glob("tokenizer_meta.json")]
    stored_type = read_json(meta_files[0])["type"] if meta_files else "char"
    if tokenizer_type is not None and stored_type != tokenizer_type:
        raise ValueError(
            f"tokenizer at {path} is of type {stored_type!r}, but configuration "
            f"expects {tokenizer_type!r}"
        )
    if stored_type == "char":
        return CharTokenizer.load(path)
    if stored_type == "hf_bpe":
        return ByteLevelBPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer type {stored_type!r} stored at {path}")


def check_vocab_compatibility(model_vocab_size: int, tokenizer: Tokenizer) -> None:
    """Fail loudly when the model's embedding matrix and tokenizer disagree."""
    if model_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model.vocab_size ({model_vocab_size}) does not match tokenizer "
            f"vocab_size ({tokenizer.vocab_size}). Either retrain the tokenizer, "
            f"or set model.vocab_size accordingly (the model must be rebuilt if "
            f"the embedding size changes)."
        )


def resolve_vocab_size(model_config, tokenizer: Tokenizer) -> int:
    """Resolve ``model.vocab_size`` against a concrete tokenizer.

    ``"auto"`` adopts the tokenizer's vocab size (recommended: model config
    files stay tokenizer-independent). An explicit integer must match.
    """
    if isinstance(model_config.vocab_size, str):
        if model_config.vocab_size != "auto":
            raise ValueError(f"invalid vocab_size {model_config.vocab_size!r}")
        return tokenizer.vocab_size
    check_vocab_compatibility(model_config.vocab_size, tokenizer)
    return model_config.vocab_size
