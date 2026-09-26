# Fontaine AI Documentation

Fontaine AI is a modular foundation-model platform: it starts as a small,
single-machine project (16 GB RAM) and scales toward large language models
without rewriting the codebase.

## Contents

| Section | What it covers |
| --- | --- |
| [architecture/](architecture/overview.md) | System overview, diagrams, component boundaries, tech stack |
| [architecture/repository-layout.md](architecture/repository-layout.md) | Every folder: responsibility, belongs / does-not-belong, evolution |
| [model/design.md](model/design.md) | Transformer components and why each exists |
| [model/tokenizer.md](model/tokenizer.md) | Tokenizer design, choices, recommendation |
| [data/pipeline.md](data/pipeline.md) | Raw → shards pipeline and memory strategy |
| [data/dataset-format.md](data/dataset-format.md) | Shard format, manifest schema, provenance/licensing |
| [training/process.md](training/process.md) | Training engine, component interfaces |
| [training/memory.md](training/memory.md) | Memory estimation methodology, 16 GB feasibility |
| [training/checkpoint-format.md](training/checkpoint-format.md) | Checkpoint contents, versioning, evolution |
| [evaluation/framework.md](evaluation/framework.md) | Evaluator registry, adding benchmarks |
| [inference/architecture.md](inference/architecture.md) | Generation engine, KV cache, sampling |
| [scaling/roadmap.md](scaling/roadmap.md) | Tiny → XL → distributed scaling plan |
| [deployment/serving.md](deployment/serving.md) | From dev server to production inference |
| [deployment/model-registry.md](deployment/model-registry.md) | Model registry design |
| [research/post-training.md](research/post-training.md) | SFT, preference optimization, RL extension points |
| [research/multimodal.md](research/multimodal.md) | Image/audio/video encoder extension points |
| [development/configuration.md](development/configuration.md) | Config hierarchy and overrides |
| [development/reproducibility.md](development/reproducibility.md) | Seeding, experiment workflow, debugging |
| [development/testing.md](development/testing.md) | Test strategy and the overfit sanity check |
| [development/security.md](development/security.md) | Security and data governance practices |
| [development/roadmap.md](development/roadmap.md) | Phases and definition of done |
| [information.html](information.html) | Single project guide: data, sizes, training, browser chat |

## Quickstart

```bash
pip install -e .            # add [bpe] for byte-level BPE tokenizers

# 1. train a tokenizer on your corpus
fontaine tokenizer train --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer --set 'data.raw_paths=["datasets/raw"]'

# 2. prepare token shards + manifest
fontaine data prepare --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer --set 'data.raw_paths=["datasets/raw"]'

# 3. train the tiny model
fontaine train --model-config configs/model/tiny.yaml \
  --training-config configs/training/local.yaml \
  --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer \
  --set data.manifest_path=datasets/prepared/manifest.json

# 4. generate — the checkpoints folder resolves newest step via latest.json
fontaine generate --checkpoint experiments/<run>/checkpoints \
  --tokenizer-dir datasets/tokenizer --prompt "Hello" --interactive
```

## The one-sentence design rule

**Build small now, architect for scale:** every subsystem is behind an
interface with exactly one production implementation today (single-machine),
and the documented evolution path replaces implementations, not code.
