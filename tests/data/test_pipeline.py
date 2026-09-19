"""End-to-end data pipeline tests: sources, cleaning, packing, manifest."""

import json

import pytest

from fontaine.config.schema import DataConfig
from fontaine.data import (
    DocumentCleaner,
    TokenShardDataset,
    prepare_dataset,
)
from fontaine.data.manifest import DatasetManifest
from fontaine.data.sources import iter_jsonl_file, iter_text_file


def _make_raw(tmp_path, documents):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "docs.txt").write_text("\n\n".join(documents), encoding="utf-8")
    return [str(raw)]


def test_iter_text_file_splits_paragraphs(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("first paragraph\nstill first\n\nsecond\n\n\nthird", encoding="utf-8")
    assert list(iter_text_file(path)) == ["first paragraph\nstill first", "second", "third"]


def test_iter_jsonl_supports_instruction_layout(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps({"text": "plain"}) + "\n"
        + json.dumps({"prompt": "q", "response": "a"}) + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    docs = list(iter_jsonl_file(path))
    assert docs[0] == "plain"
    assert docs[1] == "q\na"
    assert len(docs) == 2  # malformed line skipped with a warning


def test_cleaner_dedup_and_filters():
    config = DataConfig(raw_paths=["x"], min_document_chars=5, max_document_chars=20)
    cleaner = DocumentCleaner(config)
    docs = ["tiny", "x" * 30, "good document", "good document", "", "  good document  "]
    kept = list(cleaner.clean(iter(docs)))
    # the last two copies collapse into one (dedup) after whitespace strip
    assert kept == ["good document"]
    stats = cleaner.stats.as_dict()
    assert stats["dropped_too_short"] == 1
    assert stats["dropped_too_long"] == 1
    assert stats["dropped_empty"] == 1
    assert stats["dropped_duplicates"] == 2  # three copies -> two dropped


def test_prepare_dataset_manifest_and_splits(toy_corpus_files, tokenizer, tmp_path):
    config = DataConfig(
        raw_paths=toy_corpus_files,
        output_dir=str(tmp_path / "prepared"),
        sequence_length=32,
        val_fraction=0.2,
        min_document_chars=4,
        seed=1,
    )
    result = prepare_dataset(config, tokenizer, name="test", license="MIT")
    manifest = DatasetManifest.load(result.output_dir)

    assert manifest.name == "test" and manifest.license == "MIT"
    assert manifest.format_version == 1
    assert manifest.tokenizer["version"] == tokenizer.version
    assert set(manifest.splits) == {"train", "val"}
    assert manifest.tokens_per_split["val"] > 0
    # val_fraction respected (documents are many, so both splits exist)
    assert manifest.tokens_per_split["train"] > manifest.tokens_per_split["val"]

    manifest.validate(result.output_dir, verify_checksums=True)


def test_prepare_is_deterministic(toy_corpus_files, tokenizer, tmp_path):
    outputs = []
    for i in (1, 2):
        config = DataConfig(
            raw_paths=toy_corpus_files,
            output_dir=str(tmp_path / f"prepared_{i}"),
            sequence_length=32,
            val_fraction=0.1,
            seed=5,
        )
        result = prepare_dataset(config, tokenizer)
        manifest = DatasetManifest.load(result.output_dir)
        outputs.append([s.sha256 for s in manifest.splits["train"]])
    assert outputs[0] == outputs[1]


def test_manifest_detects_tampered_shard(toy_corpus_files, tokenizer, tmp_path):
    config = DataConfig(
        raw_paths=toy_corpus_files,
        output_dir=str(tmp_path / "prepared"),
        sequence_length=32,
        val_fraction=0.1,
    )
    result = prepare_dataset(config, tokenizer)
    shard_path = result.output_dir / "train" / DatasetManifest.load(result.output_dir).splits["train"][0].filename
    original = shard_path.read_bytes()
    shard_path.write_bytes(original[:-2] + b"\x00\x00")
    manifest = DatasetManifest.load(result.output_dir)
    with pytest.raises(ValueError, match="checksum mismatch"):
        manifest.validate(result.output_dir, verify_checksums=True)
    shard_path.write_bytes(original)  # restore for other fixtures


def test_dataset_windows_are_shifted(prepared_dir):
    dataset = TokenShardDataset(prepared_dir, "train", sequence_length=32)
    assert len(dataset) > 0
    batch = dataset[0]
    assert batch["input_ids"].shape == (32,)
    assert batch["labels"].shape == (32,)
    # labels are input shifted by one
    assert (batch["input_ids"][1:] == batch["labels"][:-1]).all()


def test_dataset_uses_lazy_memmap(prepared_dir):
    dataset = TokenShardDataset(prepared_dir, "train", sequence_length=32)
    assert dataset._memmaps == {}  # nothing opened at construction
    dataset[0]
    assert len(dataset._memmaps) >= 1  # shard opened on demand
    assert len(dataset._memmaps) <= len(dataset._shards)


def test_dataset_rejects_unknown_split(prepared_dir):
    with pytest.raises(ValueError, match="no 'test' split"):
        TokenShardDataset(prepared_dir, "test", sequence_length=32)
