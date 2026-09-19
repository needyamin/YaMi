# datasets/

Data lives here but is **never committed to Git** (see `.gitignore`).

Suggested layout:

```
datasets/
├── raw/          # your source documents (txt / jsonl / csv / json)
├── prepared/     # output of `fontaine data prepare` (shards + manifest)
└── tokenizer/    # trained tokenizer artifacts
```

Everything under this directory is reproducible from raw sources + configs,
or is an artifact with its own provenance (see `docs/data/dataset-format.md`).
