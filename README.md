# Fontaine AI

A config-driven decoder-only language model (RoPE, grouped-query attention, SwiGLU, RMSNorm) that trains on one 16 GB machine and scales by changing YAML, not the code.

Chat name: **Yami v1.0**. Full guide: [docs/information.html](docs/information.html).

## Models

Counts at an 8k vocabulary. Coding rows are mixture-of-experts: every expert is stored, and only the top experts run per token. Train with `data.sequence_length` equal to that model's context. `fontaine model inspect` prints the exact counts.

| Config | Params | Where it runs |
| --- | --- | --- |
| `tiny.yaml` | ~5.2M dense | CPU, 16 GB RAM |
| `small.yaml` | ~28M dense | modest GPU |
| `medium.yaml` | ~82M dense | free GPU (Kaggle T4) |
| `coding_low.yaml` | 5.7M active / 22M total | laptop CPU |
| `coding_mid.yaml` | 39M active / 114M total | desktop CPU or a free GPU |
| `coding_high.yaml` | 129M active / 384M total | GPU to train |

Larger sizes (0.4B and up) are the same code behind new configs. See [docs/scaling/roadmap.md](docs/scaling/roadmap.md).

## Install

```bash
pip install -e ".[bpe]"    # torch, numpy, pyyaml, byte-level BPE
pip install -e ".[dev]"    # pytest and ruff, for tests
```

CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

On Windows, if `fontaine` is not on PATH, use `python -m fontaine.cli.main <command>`.

## Chat

```bash
cp .env.example .env           # which checkpoint to serve, and the host ports
docker compose up -d web webui # chat UI → http://localhost:3000  (model: Yami v1.0)
```

Open [http://localhost:3000](http://localhost:3000). The picker shows **Yami v1.0**. The API is [http://localhost:8321](http://localhost:8321) (`/api/chat`, `/api/tags`). Set the checkpoint in `.env` (`FONTAINE_CHECKPOINT`). Details: [docs/deployment/docker.md](docs/deployment/docker.md).

## Train

Put text or JSONL in `datasets/raw`, then:

```bash
# 0. plan resources (active and total parameters, training memory)
fontaine model inspect --model-config configs/model/tiny.yaml

# 1. train a tokenizer on your corpus (hf_bpe for real runs, char for dev)
fontaine tokenizer train --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer \
  --set 'data.raw_paths=["datasets/raw"]' --set tokenizer.type=hf_bpe

# 2. validate + prepare shards + manifest
fontaine data validate --data-config configs/data/default.yaml \
  --set 'data.raw_paths=["datasets/raw"]'
fontaine data prepare --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer \
  --set 'data.raw_paths=["datasets/raw"]' --license "CC-BY-4.0"

# 3. train — must point at the manifest from step 2 (otherwise the dataset is empty)
fontaine train --model-config configs/model/tiny.yaml \
  --training-config configs/training/local.yaml \
  --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer \
  --set data.manifest_path=datasets/prepared/manifest.json

# 4. generate (streaming). checkpoints/ uses latest.json; or pass a step_ folder
fontaine generate --checkpoint experiments/<run>/checkpoints \
  --tokenizer-dir datasets/tokenizer --interactive

# 5. evaluate / serve (picker name: Yami v1.0)
fontaine evaluate --checkpoint experiments/<run>/checkpoints \
  --tokenizer-dir datasets/tokenizer
fontaine serve --checkpoint experiments/<run>/checkpoints \
  --tokenizer-dir datasets/tokenizer
```

Reference tiny run on CodeAlpaca: 5,000 steps, `val_loss` 2.14, perplexity 8.5.

No local GPU: [docs/kaggle.md](docs/kaggle.md).

## Repo

| Path | What it is |
| --- | --- |
| `src/fontaine/` | model, data, train, serve |
| `configs/` | model sizes and training |
| `tests/` | 116 CPU tests |
| `docs/` | design and how-to |
| `datasets/`, `experiments/`, `checkpoints/` | your data and weights; not committed |

## License

MIT for the code — [LICENSE](LICENSE). Your data and weights follow the license of the data you train on ([docs/development/security.md](docs/development/security.md)).
