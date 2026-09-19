"""Fontaine data pipeline: raw sources → token shards + manifest."""

from fontaine.data.cleaning import CleaningStats, DocumentCleaner
from fontaine.data.manifest import DatasetManifest, ShardInfo
from fontaine.data.pipeline import PreparationResult, prepare_dataset
from fontaine.data.shards import (
    TokenShardDataset,
    TokenShardWriter,
    build_dataloader,
    resolve_token_dtype,
)
from fontaine.data.sources import iter_documents, iter_source_documents

__all__ = [
    "DatasetManifest",
    "DocumentCleaner",
    "PreparationResult",
    "ShardInfo",
    "TokenShardDataset",
    "TokenShardWriter",
    "build_dataloader",
    "iter_documents",
    "iter_source_documents",
    "prepare_dataset",
    "resolve_token_dtype",
    "CleaningStats",
]
