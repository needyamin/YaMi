"""Dataset preparation pipeline orchestration.

    raw sources → clean/dedup/filter → tokenize → pack → shard → manifest

Packing strategy: documents are joined with the EOS token, and the token
stream is cut into non-overlapping ``sequence_length + 1`` windows (input +
shifted labels). A document-level seeded split (``val_fraction``) creates the
validation split. Peak pipeline memory = one shard buffer + one packing
window, independent of corpus size.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fontaine.config.schema import DataConfig
from fontaine.data.cleaning import CleaningStats, DocumentCleaner
from fontaine.data.manifest import DatasetManifest
from fontaine.data.shards import TokenShardWriter
from fontaine.data.sources import iter_source_documents
from fontaine.tokenizer.base import Tokenizer
from fontaine.utils.git import current_git_commit
from fontaine.utils.logging import get_logger

PREPROCESSING_VERSION = "1"

logger = get_logger("data")


@dataclass
class PreparationResult:
    """Outcome of a dataset preparation run."""

    output_dir: Path
    manifest: DatasetManifest
    cleaning_stats: CleaningStats
    tokens_dropped_remainder: int

    def summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "num_documents": self.manifest.num_documents,
            "train_tokens": self.manifest.tokens_per_split.get("train", 0),
            "val_tokens": self.manifest.tokens_per_split.get("val", 0),
            "cleaning": self.cleaning_stats.as_dict(),
            "tokens_dropped_remainder": self.tokens_dropped_remainder,
        }


def prepare_dataset(
    config: DataConfig,
    tokenizer: Tokenizer,
    name: str = "fontaine-dataset",
    dataset_version: str = "1.0.0",
    source: str = "",
    license: str | None = None,
    language: str | None = None,
    notes: str = "",
) -> PreparationResult:
    """Run the full preparation pipeline and write shards + manifest."""
    output_dir = Path(config.output_dir)
    window = config.sequence_length + 1  # input window + one shifted label

    train_writer = TokenShardWriter(
        output_dir / "train", "train", config.shard_size_tokens, tokenizer.vocab_size, config.dtype
    )
    val_writer = (
        TokenShardWriter(
            output_dir / "val", "val", config.shard_size_tokens, tokenizer.vocab_size, config.dtype
        )
        if config.val_fraction > 0
        else None
    )

    buffers: dict[str, list[int]] = {"train": [], "val": []}
    owner_buffers: dict[str, list[int]] = {"train": [], "val": []}
    doc_writers = {}
    if config.pack_documents:
        doc_writers = {
            "train": TokenShardWriter(
                output_dir / "train_docs", "train", config.shard_size_tokens, 2**31, "uint32"
            )
        }
        if val_writer is not None:
            doc_writers["val"] = TokenShardWriter(
                output_dir / "val_docs", "val", config.shard_size_tokens, 2**31, "uint32"
            )

    def push(split: str, ids: list[int], owners: list[int] | None = None) -> None:
        buffer = buffers[split]
        buffer.extend(ids)
        if owners is not None:
            owner_buffers[split].extend(owners)
        writer = train_writer if split == "train" else val_writer
        while len(buffer) >= window:
            writer.add_tokens(buffer[:window])  # type: ignore[union-attr]
            del buffer[:window]
            if split in doc_writers:
                doc_writers[split].add_tokens(owner_buffers[split][:window])
                del owner_buffers[split][:window]

    rng = random.Random(config.seed)
    cleaner = DocumentCleaner(config)
    split_rng_threshold = config.val_fraction
    if config.mixture:
        from fontaine.data.mixture import iter_mixture

        documents = iter_mixture(config.mixture, config.seed)
    else:
        documents = iter_source_documents(config.raw_paths)
    doc_index = 0

    for document in cleaner.clean(documents):
        split = (
            "val"
            if val_writer is not None and rng.random() < split_rng_threshold
            else "train"
        )
        ids = tokenizer.encode(document)
        ids.append(tokenizer.eos_id)  # EOS separates documents in the token stream
        owners = [doc_index] * len(ids) if config.pack_documents else None
        push(split, ids, owners)
        doc_index += 1

    tokens_dropped = len(buffers["train"]) + len(buffers["val"])
    if doc_writers:
        for writer in doc_writers.values():
            writer.finalize()
    train_shards = train_writer.finalize()
    val_shards = val_writer.finalize() if val_writer is not None else []

    manifest = DatasetManifest(
        name=name,
        version=dataset_version,
        source=source or ", ".join(str(p) for p in config.raw_paths),
        license=license,
        language=language,
        num_documents=cleaner.stats.kept,
        splits={"train": train_shards, **({"val": val_shards} if val_shards else {})},
        tokens_per_split={
            "train": train_writer.stats.tokens_written,
            "val": val_writer.stats.tokens_written if val_writer else 0,
        },
        cleaning_stats=cleaner.stats.as_dict(),
        tokenizer={
            "name": tokenizer.name,
            "version": tokenizer.version,
            "vocab_size": tokenizer.vocab_size,
        },
        preprocessing_version=PREPROCESSING_VERSION,
        seed=config.seed,
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=current_git_commit(),
        notes=notes,
    )
    manifest.save(output_dir)
    # Immediately verify: shards exist, counts agree, checksums match.
    manifest.validate(output_dir, verify_checksums=True)

    result = PreparationResult(
        output_dir=output_dir,
        manifest=manifest,
        cleaning_stats=cleaner.stats,
        tokens_dropped_remainder=tokens_dropped,
    )
    logger.info("dataset prepared: %s", result.summary())
    return result
