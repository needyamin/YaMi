# Fontaine AI

<img src="docs/assets/yami-v1.0.png" alt="Yami v1.0" width="72">

A config-driven decoder-only language model (RoPE, grouped-query attention, SwiGLU, RMSNorm, optional QK-norm, sliding-window attention, and mixture-of-experts) that trains on one 16 GB machine, runs on CPUs from laptops to workstations, and scales by changing YAML, not the code.

Chat name: **Yami v1.0**. Full guide: [docs/information.html](docs/information.html).

## Models

Dense rows are counted at an 8k vocabulary; coding and Yami rows use the vocabulary in their file (8k to 32k). Coding rows are mixture-of-experts: every expert is stored, and only the top experts run per token. Keep `data.sequence_length` at or below the model's context. `fontaine model inspect` prints the exact counts.

| Config | Params | Where it runs |
| --- | --- | --- |
| `tiny.yaml` | ~5.2M dense | CPU, 16 GB RAM |
| `small.yaml` | ~28M dense | modest GPU |
| `medium.yaml` | ~82M dense | free GPU (Kaggle T4) |
| `coding_low.yaml` | 5.7M active / 22M total | laptop CPU |
| `coding_mid.yaml` | 39M active / 114M total | desktop CPU or a free GPU |
| `coding_high.yaml` | 129M active / 384M total | GPU to train |
| `yami_nano.yaml` | 25M dense | any 4-core CPU, 4-8 GB RAM |
| `yami_small.yaml` | 115M dense | laptop CPU, 8-16 GB RAM |
| `yami_base.yaml` | 323M dense | 8+ core desktop, 16-32 GB RAM |
| `yami_large.yaml` | 1.06B dense | 12+ core workstation, 32 GB+ RAM |

Yami rows are counted at a 32k vocabulary. They add QK-norm, and Base and Large add sliding-window attention. The server loads them as int8 on CPU (`--precision auto`), so Large uses about 1.3 GB of weights. `fontaine model recommend` picks the tier for your machine. See [docs/scaling/roadmap.md](docs/scaling/roadmap.md).

`configs/yami_super.yaml` adds a second ladder, from a 10.6M `tiny` profile to a 1.24T-parameter mixture-of-experts `1t` profile. Those names are targets. `model estimate-params --profile <name>` prints the real counts without allocating the model. Larger than `tiny` was planned, not trained. See [docs/scaling/yami-super.md](docs/scaling/yami-super.md).

## Install

```bash
pip install -e ".[bpe]"    # torch, numpy, pyyaml, byte-level BPE
pip install -e ".[dev]"    # pytest and ruff, for tests
pip install -e ".[data]"   # optional: .parquet and .zst datasets
```

CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

On Windows, if `fontaine` is not on PATH, use `python -m fontaine.cli.main <command>`.

## Chat

```bash
cp .env.example .env        # which checkpoint to serve, and the host port
docker compose up -d web    # chat UI + API → http://localhost:8321  (model: Yami v1.0)
```

Open [http://localhost:8321](http://localhost:8321). The picker shows **Yami v1.0**. The same process serves the chat UI and the API (`/v1/chat/completions`, `/api/chat`, `/api/tags`, `/api/show`). Set the checkpoint in `.env` (`FONTAINE_CHECKPOINT`). Details: [docs/deployment/docker.md](docs/deployment/docker.md).

Serving keeps **at least 12 GiB of dense block weights resident in RAM** by default; when the model is bigger, the rest of the blocks live on disk and stream in per layer with an LRU cache and background prefetch (`GET /api/meta` reports `dense_streaming` counters). Pass `--dense-budget-mb <MiB>` to change the floor or `--dense-budget-mb 0` to keep everything resident. Smaller models are untouched.

## Train

Put data in `datasets/raw`: text, Markdown, code, JSONL/JSON/CSV/Parquet (plain, chat, or instruction records), or `.zip`/`.tar` archives of them. Formats: [docs/data/pipeline.md](docs/data/pipeline.md). Then:

```bash
# 0. plan resources (active and total parameters, training memory)
fontaine model inspect --model-config configs/model/tiny.yaml
fontaine model estimate-params --config configs/yami_super.yaml --profile 7b

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

An earlier tiny run on CodeAlpaca (5,000 steps) reached `val_loss` 2.14, perplexity 8.5. The only run in `experiments/` today is a 3.2M toy quickstart model, so chat replies are empty or random words until you train on real data. `serve` and `generate` take `--precision` (default `auto`: int8 on CPU from 20M parameters) and `--threads`.

No local GPU: [docs/kaggle.md](docs/kaggle.md).

## Repo

| Path | What it is |
| --- | --- |
| `src/fontaine/` | model, data, train, serve |
| `configs/` | model sizes and training |
| `tests/` | 211 CPU tests |
| `docs/` | design and how-to |
| `datasets/`, `experiments/`, `checkpoints/` | your data and weights; not committed |

## License

MIT for the code — [LICENSE](LICENSE). Your data and weights follow the license of the data you train on ([docs/development/security.md](docs/development/security.md)).
