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
| Reading | `sources.py`, `records.py` | Files, directories, and archives; see "Supported raw data" below. Everything streams, and nothing is extracted to disk. |
| Validation | `cli data validate` | readability, format errors, document/char stats; manifest mode verifies SHA-256 checksums |
| Cleaning / filtering / normalization | `cleaning.py` | NFKC unicode normalize, whitespace strip, min/max char filters, empty-drop |
| Deduplication | `cleaning.py` | exact-document dedup via 16-byte BLAKE2b fingerprints (memory-bounded) |
| Tokenization | `pipeline.py` | batched encode per document; EOS appended as document separator |
| Packing | `pipeline.py` | token stream cut into non-overlapping `sequence_length + 1` windows (input + shifted labels); windows may span documents across EOS |
| Sharding | `shards.py` | fixed-size raw uint16/uint32 `.bin` files; one buffer in memory |
| Manifest | `manifest.py` | metadata + per-shard checksums; validated (checksums recomputed) before the run is accepted |

A document-level seeded split (`data.val_fraction`) carves out validation
shards, so evaluation never sees training documents.

## Supported raw data

Point `data.raw_paths` at any mix of files, folders, and archives.

| Kind | Extensions | One document is |
| --- | --- | --- |
| Plain text | `.txt`, `.text` | a paragraph (split on blank lines) |
| Prose and markup | `.md`, `.rst`, `.tex`, `.html`, `.xml`, `.srt`, ... | the whole file |
| Source code | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.sql`, `.sh`, `.yaml`, `Dockerfile`, ... | the whole file |
| Records | `.jsonl`, `.ndjson`, `.json`, `.csv`, `.tsv`, `.parquet` | one record |
| Notebooks | `.ipynb` | the markdown and code cells |
| Archives | `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz` | each member, read by its own extension; nested archives work |
| Compression | `.gz`, `.bz2`, `.xz`, `.zst` on any file above | the inner file |

Records can use any of these layouts (field names are case-insensitive):

- `text`, or another text field: `content`, `body`, `code`, `document`, ...
- chat: `messages` / `conversations` with `{role, content}` (OpenAI) or
  `{from, value}` (ShareGPT);
- instruction: `instruction` / `prompt` / `question` plus `output` /
  `response` / `completion` / `answer`, with an optional Alpaca `input` and
  `system`.

Chat and instruction records are rendered in the same Alpaca layout that
`/api/chat` builds at serve time (`### Instruction:` / `### Response:`), so
the model learns the format it is prompted with. JSON files wrapped as
`{"data": [...]}` are unwrapped.

Hidden files, `__MACOSX/` folders, encrypted zip members, and unsupported
types (images, binaries) are skipped with a warning. Text is decoded as UTF-8
(BOM allowed) or BOM-marked UTF-16. `.parquet` needs `pyarrow` and `.zst`
needs `zstandard`: `pip install -e ".[data]"` (the Docker image has both).

Use the byte-level BPE tokenizer (`tokenizer.type=hf_bpe`) for mixed data.
It encodes any language, symbol, or code without unknown tokens. The char
tokenizer maps characters it did not see in training to `<|unk|>`.

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
- code: a code-aware cleaner (license headers, minified-file filter).
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
