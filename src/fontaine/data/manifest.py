"""Dataset manifest: provenance, metadata, and integrity for prepared data.

The manifest is the single source of truth a training run consumes. It
records *what* the data is, *where* it came from, *how* it was processed
(tokenizer + preprocessing versions), and checksums for every shard so
training can verify integrity. Format spec: ``docs/data/dataset-format.md``.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fontaine.utils.hashing import sha256_file
from fontaine.utils.io import atomic_write_json

MANIFEST_FILENAME = "manifest.json"
MANIFEST_FORMAT_VERSION = 1


@dataclass
class ShardInfo:
    """One binary token shard: a raw array of unsigned ints (memmap-able)."""

    filename: str
    split: str  # "train" | "val"
    num_tokens: int
    sha256: str


@dataclass
class DatasetManifest:
    """Complete metadata for a prepared dataset."""

    name: str
    version: str
    format_version: int = MANIFEST_FORMAT_VERSION
    # Provenance / governance.
    source: str = ""
    license: str | None = None
    language: str | None = None
    # Content stats.
    num_documents: int = 0
    splits: dict[str, list[ShardInfo]] = field(default_factory=dict)
    tokens_per_split: dict[str, int] = field(default_factory=dict)
    cleaning_stats: dict[str, int] = field(default_factory=dict)
    # Reproducibility.
    tokenizer: dict[str, object] = field(default_factory=dict)  # name/version/vocab_size
    preprocessing_version: str = ""
    seed: int = 0
    created_utc: str = ""
    git_commit: str = "unknown"
    notes: str = ""

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / MANIFEST_FILENAME
        atomic_write_json(path, asdict(self))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "DatasetManifest":
        path = Path(directory) / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no dataset manifest at {path} — run 'fontaine data prepare' first"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["splits"] = {
            split: [ShardInfo(**shard) for shard in shards]
            for split, shards in payload.get("splits", {}).items()
        }
        payload.pop("shards", None)  # forward-compat: ignore unknown legacy keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in payload.items() if k in known})

    # -- integrity -----------------------------------------------------------

    def validate(self, directory: str | Path, verify_checksums: bool = False) -> None:
        """Check shard presence/counts; optionally verify SHA-256 per shard."""
        directory = Path(directory)
        if self.format_version != MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"manifest format_version {self.format_version} != supported "
                f"{MANIFEST_FORMAT_VERSION}"
            )
        if not self.splits:
            raise ValueError("manifest contains no splits")
        for split, shards in self.splits.items():
            total = 0
            for shard in shards:
                shard_path = directory / split / shard.filename
                if not shard_path.is_file():
                    raise FileNotFoundError(f"missing shard file: {shard_path}")
                total += shard.num_tokens
                if verify_checksums:
                    actual = sha256_file(shard_path)
                    if actual != shard.sha256:
                        raise ValueError(
                            f"checksum mismatch for {shard_path}: "
                            f"manifest={shard.sha256} actual={actual}"
                        )
            expected = self.tokens_per_split.get(split)
            if expected is not None and expected != total:
                raise ValueError(
                    f"split '{split}': token count mismatch "
                    f"(manifest={expected}, shards={total})"
                )

    @property
    def num_tokens(self) -> int:
        return sum(self.tokens_per_split.values())
