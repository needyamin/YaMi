# Security & Data Governance

**Important framing:** the engineering practices below reduce *risk*; they are
not legal advice and make no compliance claims. Licensing questions (fair use,
training-data rights, output obligations) require qualified legal counsel in
your jurisdiction.

## Untrusted datasets

- Data is parsed defensively: malformed JSONL lines are skipped with warnings
  (`errors="replace"` decoding, no crashes on bad bytes); JSON/CSV parsers are
  the stdlib's.
- Cleaning filters bound document sizes *before* tokenization — hostile
  pathological inputs (billions of repeated characters) are dropped by
  `max_document_chars`.
- Manifest checksums detect accidental or malicious corruption of prepared
  shards; `fontaine data validate` and checkpoint load paths verify.

## Malicious input / prompt injection in training data

- Training text shapes model behavior; a corpus can carry instructions aimed
  at future users ("ignore previous instructions…"). Engineering mitigations:
  source provenance per document (manifest), inspectability of the prepared
  stream (plain binary + manifest — easy to sample and audit), and evaluation
  on held-out safety/benchmark suites before deployment.
- Runtime: the dev server binds localhost by default and performs no eval of
  user content; treat any deployment beyond localhost as a production system
  needing auth, rate limiting, and output filtering (`docs/deployment/serving.md`).

## Copyrighted data, personal information, dataset/model licenses

- **Bookkeeping (implemented):** manifests record `source`, `license`,
  `language`, and creation metadata; checkpoints and experiments propagate
  dataset identity forward, so every model is traceable to its data.
- **Process (your responsibility):** decide what you may use. Prefer
  permissively licensed / public-domain / self-authored data; keep raw-source
  records; honor dataset licenses that impose redistribution or attribution
  terms. This project ships MIT *code* — it says nothing about your *data*.
- **PII:** no PII scanning ships in v1. If your data may contain personal
  information, run an external scanner on raw data *before* `data prepare`,
  and record the audit outcome in the manifest `notes`.

## Model provenance & checkpoint integrity

- Checkpoints are hash-sealed (SHA-256 per file, verified on load) and carry
  config/tokenizer/dataset/code versions — a model's identity is verifiable.
- `torch.load(weights_only=True)` semantics are preserved (RNG states are
  stored in plain-int form) to minimize pickle attack surface; never load
  checkpoints from untrusted sources.
- Registry status transitions (`candidate → evaluated → deployed`) should gate
  deployment on recorded evaluation, not on vibes.

## Dependency security

- Runtime dependencies are three (torch, numpy, pyyaml); optional extras are
  isolated to single modules.
- Pin and audit: `pip install -r requirements/requirements.txt` with your own
  pinning/lock layer; run `pip-audit` in CI when available.
- No network access is required at training/inference time — data and weights
  are local, and the codebase performs no telemetry.

## Security-relevant defaults

| Default | Rationale |
| --- | --- |
| dev server binds `127.0.0.1` | no accidental exposure |
| integrity verification on checkpoint load | corruption/actor detection |
| checksummed dataset manifests | silent-data-corruption detection |
| `-100` label masking contract | no accidental label leakage in SFT |
