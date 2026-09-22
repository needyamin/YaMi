# Running Fontaine in Docker

Everything — training, serving, and the web playground — runs from one CPU
image. Model artifacts (tokenizer, prepared data, checkpoints) are **not**
baked into the image; they live on the host and are bind-mounted, so a new
training run never requires a rebuild.

## Files

| File | Role |
| --- | --- |
| `Dockerfile` | CPU image: PyTorch (CPU wheels) + `fontaine-ai .[bpe]`, unprivileged user, health check |
| `docker-compose.yml` | `web` (playground + API, default) and `train` (one-off run, `train` profile) |
| `.dockerignore` | keeps data/models/caches out of the build context |
| `.env.example` | template for choosing the served checkpoint and host port |

## Quickstart (serve)

```bash
cp .env.example .env        # then edit FONTAINE_CHECKPOINT if needed
docker compose up -d web    # build + start (first build downloads ~1 GB)
```

Open **http://localhost:8321** — the playground. Type an instruction, press
Enter; the UI wraps it in the `### Instruction / ### Response` template the
model was trained on and streams the answer.

Programmatic API (same contract as `fontaine serve`):

```bash
curl http://localhost:8321/health
curl -X POST http://localhost:8321/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "### Instruction:\nSay hi.\n\n### Response:\n", "stream": false, "max_new_tokens": 64}'
```

## Train inside Docker

Prepare data first on the host (tokenizer + `fontaine data prepare`, see
[docs/info.html](../info.html) for the gentle version), then:

```bash
docker compose run --rm train
```

The run writes `datasets/prepared/` and `experiments/<timestamp>_<run>/` back
to the host folders. When it finishes, point the web service at the new run:

```bash
# .env
FONTAINE_CHECKPOINT=/app/experiments/<new-run-folder>/checkpoints
```

```bash
docker compose up -d web    # recreates the container with the new checkpoint
```

`checkpoints` (the folder, not a step inside it) always resolves to the newest
saved step via its `latest.json` pointer.

## Operations

| Task | Command |
| --- | --- |
| Logs | `docker compose logs -f web` |
| Stop | `docker compose down` |
| Rebuild after source changes | `docker compose up -d --build web` |
| Different host port | set `FONTAINE_PORT` in `.env` |
| Manually pick a step | `FONTAINE_CHECKPOINT=/app/experiments/<run>/checkpoints/step_00003000` |

## Security notes

- The container runs as an unprivileged user; no shell packages beyond the
  Python standard library are installed.
- The server binds `0.0.0.0` **inside** the container so the published port
  works; publish only to `127.0.0.1` if you edit `ports`
  (`127.0.0.1:8321:8321`) and do not want other machines on your LAN to reach
  the playground.
- This is the development server (stdlib `http.server`) with no auth — see
  [serving.md](serving.md) for the production path. Do not expose it to the
  internet.
