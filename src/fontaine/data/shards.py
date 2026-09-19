"""Binary token shards and memory-mapped dataset access.

Prepared data lives as raw arrays of unsigned ints (``.bin`` files) that can
be opened with ``np.memmap``: the OS pages tokens in on demand, so a dataset
many times larger than RAM trains without loading it. One shard is written at
a time from a single fixed-size buffer — the pipeline's peak memory is one
shard buffer plus one packing window, independent of corpus size.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fontaine.data.manifest import ShardInfo

SHARD_SUFFIX = ".bin"
_UINT16_MAX = 65_535


def resolve_token_dtype(vocab_size: int, requested: str = "auto") -> np.dtype:
    """Pick the on-disk token dtype; 'auto' minimizes bytes per token."""
    if requested == "uint16":
        if vocab_size > _UINT16_MAX + 1:
            raise ValueError(f"vocab_size {vocab_size} does not fit uint16 shards")
        return np.dtype("uint16")
    if requested == "uint32":
        return np.dtype("uint32")
    if requested == "auto":
        return np.dtype("uint16" if vocab_size <= _UINT16_MAX + 1 else "uint32")
    raise ValueError(f"unknown dtype {requested!r} (expected auto/uint16/uint32)")


@dataclass
class ShardWriterStats:
    shards_written: int = 0
    tokens_written: int = 0


class TokenShardWriter:
    """Streams tokens into fixed-size ``.bin`` shards under a split directory.

    Files are named ``<split>_<index:05d>.bin`` and are exactly
    ``shard_size_tokens * dtype.itemsize`` bytes (except the final partial
    shard), which makes random access via memmap trivial.
    """

    def __init__(
        self,
        directory: str | Path,
        split: str,
        shard_size_tokens: int,
        vocab_size: int,
        dtype: str = "auto",
    ) -> None:
        self.directory = Path(directory)
        self.split = split
        self.shard_size_tokens = shard_size_tokens
        self.dtype = resolve_token_dtype(vocab_size, dtype)
        self.stats = ShardWriterStats()
        self._records: list[ShardInfo] = []
        self._buffer = np.empty(shard_size_tokens, dtype=self.dtype)
        self._count = 0
        self._shard_index = 0

    def add_tokens(self, ids: Iterable[int]) -> None:
        """Append token ids; flushes full shards automatically."""
        if isinstance(ids, np.ndarray):
            self._add_array(ids)
            return
        for token in ids:
            if not 0 <= int(token) < np.iinfo(self.dtype).max + 1:
                raise ValueError(f"token id {token} out of range for dtype {self.dtype}")
            self._buffer[self._count] = token
            self._count += 1
            if self._count == self.shard_size_tokens:
                self._flush()

    def _add_array(self, array: np.ndarray) -> None:
        array = np.asarray(array, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() > np.iinfo(self.dtype).max):
            raise ValueError("token id out of range for shard dtype")
        offset = 0
        while offset < array.size:
            space = self.shard_size_tokens - self._count
            chunk = array[offset : offset + space]
            self._buffer[self._count : self._count + chunk.size] = chunk
            self._count += chunk.size
            offset += chunk.size
            if self._count == self.shard_size_tokens:
                self._flush()

    def _flush(self) -> None:
        if self._count == 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"{self.split}_{self._shard_index:05d}{SHARD_SUFFIX}"
        path = self.directory / filename
        # tofile writes the exact bytes of the view — no full-file RAM buffer.
        self._buffer[: self._count].tofile(path)
        from fontaine.utils.hashing import sha256_file

        self._records.append(
            ShardInfo(
                filename=filename,
                split=self.split,
                num_tokens=self._count,
                sha256=sha256_file(path),
            )
        )
        self.stats.shards_written += 1
        self.stats.tokens_written += self._count
        self._count = 0
        self._shard_index += 1

    def finalize(self) -> list[ShardInfo]:
        """Flush the partial shard and return all shard records for the manifest."""
        self._flush()
        return list(self._records)


class TokenShardDataset(Dataset):
    """Random-access fixed windows over memory-mapped token shards.

    Window ``i`` holds ``sequence_length + 1`` consecutive tokens; ``__getitem__``
    returns ``input_ids = window[:-1]`` and ``labels = window[1:]`` (the standard
    next-token shift). The window index → (shard, offset) map is built purely
    from manifest metadata, so construction reads no token data at all.
    """

    def __init__(self, directory: str | Path, split: str, sequence_length: int) -> None:
        from fontaine.data.manifest import DatasetManifest

        self.directory = Path(directory)
        self.split = split
        self.sequence_length = sequence_length
        self.manifest = DatasetManifest.load(self.directory)
        self.manifest.validate(self.directory, verify_checksums=False)
        if split not in self.manifest.splits:
            raise ValueError(f"manifest has no '{split}' split (has: {list(self.manifest.splits)})")

        self._shards = self.manifest.splits[split]
        self._dtype = resolve_token_dtype(
            int(self.manifest.tokenizer["vocab_size"]), "auto"
        )
        # Compact index: (shard_idx, start) per window.
        self._index: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self._shards):
            windows = (shard.num_tokens - 1) // sequence_length
            for w in range(windows):
                self._index.append((shard_idx, w * sequence_length))
        self._memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _open_memmap(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._memmaps:
            path = self.directory / self.split / self._shards[shard_idx].filename
            self._memmaps[shard_idx] = np.memmap(path, dtype=self._dtype, mode="r")
        return self._memmaps[shard_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx, start = self._index[idx]
        window = self._open_memmap(shard_idx)[start : start + self.sequence_length + 1]
        tokens = torch.from_numpy(window.astype(np.int64))
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_dataloader(
    dataset: TokenShardDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Build a DataLoader; uses a distributed sampler when running multi-process."""
    from torch.utils.data.distributed import DistributedSampler

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    sampler = None
    shuffle_arg = shuffle
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle_arg = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_arg,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
