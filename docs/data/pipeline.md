# Data Pipeline

```
Raw Data → Validation → Cleaning → Deduplication → Filtering → Normalization
        → Tokenization → Packing → Sharding → Dataset Manifest → Training
```

Implemented in `src/fontaine/data/` and driven by
`fontaine data prepare` (`pipeline.prepare_dataset`).

## Stage by stage

| Stage | Module | Notes |
| --- | --- | --- |
| Reading | `sources.py` | `.txt` (paragraph-split), `.jsonl` (one doc/line; `text` or `prompt`+`response`), `.json` (array; loads fully — prefer JSONL), `.csv` (text column). Directories are walked recursively. Everything streams. |
| Validation | `cli data validate` | readability, format errors, document/char stats; manifest mode verifies SHA-256 checksums |
| Cleaning / filtering / normalization | `cleaning.py` | NFKC unicode normalize, whitespace strip, min/max char filters, empty-drop |
| Deduplication | `cleaning.py` | exact-document dedup via 16-byte BLAKE2b fingerprints (memory-bounded) |
| Tokenization | `pipeline.py` | batched encode per document; EOS appended as document separator |
| Packing | `pipeline.py` | token stream cut into non-overlapping `sequence_length + 1` windows (input + shifted labels); windows may span documents across EOS |
| Sharding | `shards.py` | fixed-size raw uint16/uint32 `.bin` files; one buffer in memory |
| Manifest | `manifest.py` | metadata + per-shard checksums; validated (checksums recomputed) before the run is accepted |

A document-level seeded split (`data.val_fraction`) carves out validation
shards, so evaluation never sees training documents.

## How huge datasets avoid huge RAM

1. **Streaming end to end** — readers yield one document at a time; the
   cleaner is an iterator transform; the tokenizer consumes streams.
2. **Fixed shard buffer** — `TokenShardWriter` holds exactly one
   `shard_size_tokens` buffer; shards are written with `ndarray.tofile`
   (no full-file buffer).
3. **Memory-mapped reads** — `TokenShardDataset` opens shards with
   `np.memmap`; the OS pages tokens in on demand. A 50 GB dataset trains
   comfortably in 16 GB RAM.
4. **Bounded dedup memory** — 16 bytes per unique document
   (web-scale corpora move to Bloom-filter/MinHash external tools —
   same interface).
5. **Peak pipeline memory** ≈ one shard buffer + one packing window +
   the dedup fingerprint set — independent of corpus size.

## Future dataset types

The reader layer is the only extension point:
- conversations / chat: JSONL with `messages` — flatten or serialize with
  speaker markers (a `conversation` reader).
- code: typically plain text with language-tagged provenance; a code-aware
  cleaner (license headers, minified-file filter).
- multimodal: pairs (text, media-path) — encoders attach at the *model*
  layer (`docs/research/multimodal.md`); the token-shard contract gains a
  parallel media-shard format.

## Provenance & licensing (engineering practice)

`DatasetManifest` records `source`, `license`, `language`, `version`,
`preprocessing_version`, `tokenizer version`, `git_commit`, `seed`,
`created_utc`, per-shard SHA-256, and cleaning statistics. `fontaine data
prepare --license SPDX-ID --source …` records them at creation time.

This is bookkeeping, not legal clearance: see
`docs/development/security.md` for the engineering/legal distinction.
