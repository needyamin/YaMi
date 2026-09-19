# experiments/

One directory per training run — **never committed to Git**.

```
experiments/<date>_<time>_<run_name>/
├── config.yaml        # resolved configuration snapshot
├── environment.json   # git commit, versions, device, seed
├── metrics.jsonl      # step-ordered metrics
├── summary.json       # final metrics + duration
├── logs/
└── checkpoints/
```

Contract: `docs/development/reproducibility.md`.
