"""Shared fixtures: a tiny synthetic corpus, tokenizer, model config, and a
prepared dataset — built once per session so the whole suite stays fast."""

from pathlib import Path

import pytest

from fontaine.config.schema import DataConfig, ModelConfig
from fontaine.data import prepare_dataset
from fontaine.tokenizer import CharTokenizer

# Highly predictable text: a tiny model can genuinely overfit this, which is
# the sanity check the training test relies on.
_CORPUS_TEMPLATE = (
    "the fontaine flows over the stone {n} and the water sings\n"
    "the water sings and the fontaine dreams\n\n"
    "line {n} of the fontaine song is about water and stone\n"
    "stone and water and fontaine song {n}\n\n"
)


@pytest.fixture(scope="session")
def toy_corpus() -> list[str]:
    return [_CORPUS_TEMPLATE.format(n=i) for i in range(40)]


@pytest.fixture(scope="session")
def toy_corpus_files(toy_corpus: list[str], tmp_path_factory) -> list[str]:
    raw_dir = tmp_path_factory.mktemp("raw")
    path = raw_dir / "corpus.txt"
    path.write_text("".join(toy_corpus), encoding="utf-8")
    return [str(path)]


@pytest.fixture(scope="session")
def tokenizer(toy_corpus: list[str], tmp_path_factory) -> CharTokenizer:
    tok = CharTokenizer.train(toy_corpus)
    tok.save(tmp_path_factory.mktemp("tokenizer"))
    return tok


@pytest.fixture(scope="session")
def model_config(tokenizer: CharTokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=192,
        max_sequence_length=64,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def prepared_dir(toy_corpus_files: list[str], tokenizer: CharTokenizer, tmp_path_factory) -> Path:
    """A fully prepared dataset (shards + manifest) built once per session."""
    config = DataConfig(
        raw_paths=toy_corpus_files,
        output_dir=str(tmp_path_factory.mktemp("prepared")),
        sequence_length=32,
        shard_size_tokens=50_000,
        val_fraction=0.1,
        min_document_chars=4,
        seed=123,
    )
    result = prepare_dataset(config, tokenizer, name="toy-dataset")
    return result.output_dir
