# Repository Layout

```
fontaine-ai/
├── README.md
├── LICENSE
├── pyproject.toml
├── read.txt                  # original specification (local)
├── requirements/
│   ├── requirements.txt      # runtime deps (torch, numpy, pyyaml)
│   └── requirements-dev.txt  # + pytest, tokenizers, ruff
├── configs/
│   ├── model/                # tiny.yaml · small.yaml · medium.yaml
│   ├── training/             # local.yaml · tiny_test.yaml
│   ├── data/                 # default.yaml
│   └── inference/            # default.yaml
├── docs/                     # see docs/index.md
├── scripts/                  # convenience entry points
├── src/fontaine/
│   ├── models/               # architecture + components + KV cache
│   ├── tokenizer/            # Tokenizer interface, char + BPE impls
│   ├── data/                 # sources, cleaning, shards, manifest, pipeline
│   ├── training/             # trainer, optimizer/scheduler, experiments
│   ├── evaluation/           # evaluator registry + loss evaluator
│   ├── inference/            # generator, checkpoint loader, dev server
│   ├── generation/           # sampling strategies (pure functions)
│   ├── distributed/          # ParallelContext + TrainingStrategy seams
│   ├── checkpointing/        # CheckpointManager, converter
│   ├── optimization/         # memory estimator, device/precision resolution
│   ├── utils/                # io, hashing, seeding, logging, git
│   ├── config/               # typed schemas, YAML loader, overrides
│   └── cli/                  # argparse front-end (fontaine …)
├── tests/
│   ├── unit/                 # config, sampling, memory, utils
│   ├── model/                # components + forward/backward
│   ├── tokenizer/
│   ├── data/
│   ├── training/             # trainer, checkpointing
│   ├── inference/
│   └── integration/          # full CLI pipeline test
├── datasets/                 # gitignored — raw + prepared data only
├── checkpoints/              # gitignored — standalone checkpoints
├── experiments/              # gitignored — one dir per run
├── logs/                     # gitignored
└── tools/                    # one-off maintenance utilities
```

## Directory contracts

### `src/fontaine/` — source of truth

| Package | Responsibility | Belongs here | Does NOT belong here | How it evolves |
| --- | --- | --- | --- | --- |
| `models/` | Architecture definition | layers, attention, KV cache, factory | training logic, data code, generation loops | new architectures register in `factory.py` |
| `tokenizer/` | Text ↔ ids | interface, implementations, registry | model code, dataset format | new tokenizer types as new classes |
| `data/` | Raw → shards | readers, cleaning, packing, manifest, dataset | model/training logic | object-storage backends, distributed workers |
| `training/` | The engine | `Trainer`, optimizer/scheduler, experiment dirs | model math, data formats, generation | SFT/post-training trainers reuse the same collaborators |
| `evaluation/` | Benchmarks | registry, evaluators | training-loop changes | generative benchmarks subclass `GenerativeBenchmark` |
| `inference/` | Serving generation | engine, loader, dev server | sampling math (lives in `generation/`) | production server behind same `Generator` |
| `generation/` | Sampling | pure token-sampling functions | model code, HTTP | batched/logits-processor strategies |
| `distributed/` | Parallelism seams | context, strategies | eager infra | DDP→FSDP→TP/PP implementations |
| `checkpointing/` | Artifact persistence | manager, format, converter | what to save (trainer decides) | sharded rank layouts, cloud upload |
| `optimization/` | Resource planning | memory estimator, precision plan | training policy | offloading helpers, quantization wrappers |
| `config/` | Typed configuration | dataclasses, loader | hidden defaults outside schema | additive fields only |
| `utils/`, `cli/` | Shared plumbing | — | business logic | — |

### Data and artifacts (never committed to Git)

- `datasets/` — raw inputs and prepared shards. Large, reproducible from raw
  sources + configs. `.gitignore`d except a README.
- `checkpoints/` — standalone checkpoints outside experiments. Binary, large.
- `experiments/` — per-run artifacts (config snapshot, metrics, logs,
  checkpoints). Valuable but regenerable; too big for Git.
- `logs/` — throwaway log output.
- `tests/` — contains a tiny synthetic dataset *generator* (not data files),
  so the pipeline is tested without committing any data.

### Why source/artifact separation matters

1. Git history stays small and reviewable; diffs are code, not binaries.
2. Checkpoints/datasets have their own lifecycle (retention, pruning,
   provenance) managed by manifests and `CheckpointManager`, not Git.
3. CI can clone the repo and run the full test suite in minutes without any
   dataset or model download.
