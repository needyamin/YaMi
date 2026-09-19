# Dataset Format Specification

A **prepared dataset** is a directory:

```
datasets/prepared/
├── manifest.json
├── train/
│   ├── train_00000.bin      # raw token array, uint16/uint32, little-endian
│   ├── train_00001.bin
│   └── …
└── val/
    └── …
```

## Shards

- Each `.bin` file is a flat C-order array of unsigned ints
  (`uint16` when `vocab_size ≤ 65536`, else `uint32` — `data.dtype: auto`).
- Shard capacity is `data.shard_size_tokens`; only the final shard of a
  split may be partial.
- Tokens are the packed stream: documents joined with `eos_id`, cut into
  `sequence_length + 1` windows at read time (`input = window[:-1]`,
  `labels = window[1:]`).
- Files are opened with `np.memmap` — no load step, random access,
  OS-managed paging.
- Exactly one token per shard boundary is sacrificed per window alignment
  (`(num_tokens - 1) // sequence_length` windows per shard); remainders are
  counted in the manifest.

## Manifest schema (`manifest.json`, format_version 1)

| Field | Type | Meaning |
| --- | --- | --- |
| `name`, `version` | string | human identity (e.g. `fontaine-corpus` `1.0.0`) |
| `format_version` | int | manifest schema version (1) |
| `source` | string | origin description / input paths |
| `license` | string? | SPDX-style identifier or license note |
| `language` | string? | primary language(s) |
| `num_documents` | int | documents kept after cleaning |
| `splits` | `{split: [ShardInfo]}` | per-split shard list |
| `tokens_per_split` | `{split: int}` | totals (cross-checked on validate) |
| `cleaning_stats` | object | seen/kept/dropped counters |
| `tokenizer` | `{name, version, vocab_size}` | exact tokenizer identity |
| `preprocessing_version` | string | pipeline code version that produced it |
| `seed` | int | split RNG seed |
| `created_utc` | ISO timestamp | creation time |
| `git_commit` | string | pipeline provenance |
| `notes` | string | free-form |

`ShardInfo = {filename, split, num_tokens, sha256}`.

`manifest.validate(directory, verify_checksums=True)` checks schema version,
shard presence, token totals, and recomputes every SHA-256 —
`fontaine data validate` runs exactly this.

## Provenance and licensing considerations

- Every derived artifact (manifests, checkpoints, experiment configs) carries
  the dataset name/version and tokenizer version — a model is always
  traceable to its data.
- Record the license **at preparation time** (`--license`). Prefer
  machine-readable SPDX identifiers. Keep raw-source retention notes in
  `--notes`.
- Data governance (PII scanning, takedown handling) is a process layered on
  the pipeline: run it on raw data, record the outcome in `notes`, and keep
  the manifest as the audit anchor. See `docs/development/security.md`.

## Evolution

- `format_version` gates compatibility: loaders refuse unknown versions
  instead of misreading bytes.
- Future: parallel media shards (multimodal), per-shard metadata blocks,
  object-storage URLs in `ShardInfo` — additive fields, same schema pattern.
