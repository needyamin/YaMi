# Running Fontaine in Docker

Training, serving, and the chat UI run from one CPU image. Model artifacts
(tokenizer, prepared data, checkpoints) are **not** baked into the image; they
live on the host and are bind-mounted, so a new training run never requires a
rebuild. The chat UI is built into the image and served by the same process
as the API.

## Files

| File | Role |
| --- | --- |
| `Dockerfile` | CPU image: chat UI build, PyTorch (CPU wheels) + `fontaine-ai .[bpe]`, unprivileged user, health check |
| `docker-compose.yml` | `web` (chat UI and API, default), `train` (one-off run, `train` profile) |
| `.dockerignore` | keeps data/models/caches out of the build context |
| `.env.example` | template for choosing the served checkpoint and host port |

## Quickstart (serve + chat)

```bash
cp .env.example .env        # then edit FONTAINE_CHECKPOINT if needed
docker compose up -d web    # first start builds the CPU image, including the chat UI
```

Open **http://localhost:8321**. The model picker shows **Yami v1.0**
(the name in `configs/inference/default.yaml`). Chat uses
`POST /v1/chat/completions`. Curl can use that route or the Ollama-compatible
`/api/chat`.

Programmatic API:

```bash
curl http://localhost:8321/api/version
curl http://localhost:8321/v1/models
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hi."}], "stream": false, "max_completion_tokens": 64}'
curl -X POST http://localhost:8321/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hi."}], "stream": false, "options": {"num_predict": 64}}'
curl -X POST http://localhost:8321/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "### Instruction:\nSay hi.\n\n### Response:\n", "stream": false, "max_new_tokens": 64}'
```

## Train inside Docker

Prepare data first on the host (tokenizer + `fontaine data prepare`, see
[docs/information.html](../information.html)), then:

```bash
docker compose run --rm train
```

The run reads the prepared manifest and writes
`experiments/<timestamp>_<run>/` (metrics, logs, checkpoints) back to the
host. When it finishes, point the web service at the new run:

```bash
# .env
FONTAINE_CHECKPOINT=/app/experiments/<new-run-folder>/checkpoints
```

```bash
docker compose up -d web   # recreates the container with the new checkpoint
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

- The Fontaine container runs as an unprivileged user; no shell packages
  beyond the Python standard library are installed.
- The server binds `0.0.0.0` **inside** the container so the published port
  works; publish only to `127.0.0.1` if you edit `ports`
  (`127.0.0.1:8321:8321`) and do not want other machines on your LAN to reach
  the API.
- This is the development server (stdlib `http.server`) with no auth — see
  [serving.md](serving.md) for the production path. Do not expose port 8321
  to the internet.
