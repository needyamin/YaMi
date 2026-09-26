# Running Fontaine in Docker

Everything — training, serving, and the chat UI — runs from one CPU image
plus the Open WebUI container. Model artifacts (tokenizer, prepared data,
checkpoints) are **not** baked into the image; they live on the host and are
bind-mounted, so a new training run never requires a rebuild.

## Files

| File | Role |
| --- | --- |
| `Dockerfile` | CPU image: PyTorch (CPU wheels) + `fontaine-ai .[bpe]`, unprivileged user, health check |
| `docker-compose.yml` | `web` (Ollama-compatible API, default), `webui` (Open WebUI chat interface), `train` (one-off run, `train` profile) |
| `.dockerignore` | keeps data/models/caches out of the build context |
| `.env.example` | template for choosing the served checkpoint and host ports |

## Quickstart (serve + chat)

```bash
cp .env.example .env        # then edit FONTAINE_CHECKPOINT if needed
docker compose up -d web webui   # first start pulls ~1 GB + the Open WebUI image
```

Open **http://localhost:3000** — Open WebUI, the open-source Ollama-style
chat interface. The model picker shows **Yami v1.0** (the name in
`configs/inference/default.yaml`). Type an instruction and the answer
streams back. (The `web` service speaks the Ollama API at
http://localhost:8321 — any Ollama-compatible UI or client can connect the
same way.)

Programmatic API:

```bash
curl http://localhost:8321/api/version
curl http://localhost:8321/api/tags
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
docker compose up -d web webui   # recreates the containers with the new checkpoint
```

`checkpoints` (the folder, not a step inside it) always resolves to the newest
saved step via its `latest.json` pointer.

## Operations

| Task | Command |
| --- | --- |
| Logs | `docker compose logs -f web` / `docker compose logs -f webui` |
| Stop | `docker compose down` |
| Rebuild after source changes | `docker compose up -d --build web webui` |
| Different host ports | set `FONTAINE_PORT` / `WEBUI_PORT` in `.env` |
| Manually pick a step | `FONTAINE_CHECKPOINT=/app/experiments/<run>/checkpoints/step_00003000` |

## Security notes

- The Fontaine container runs as an unprivileged user; no shell packages
  beyond the Python standard library are installed.
- The server binds `0.0.0.0` **inside** the container so the published port
  works; publish only to `127.0.0.1` if you edit `ports`
  (`127.0.0.1:8321:8321`) and do not want other machines on your LAN to reach
  the API.
- This is the development server (stdlib `http.server`) with no auth — see
  [serving.md](serving.md) for the production path. Do not expose it to the
  internet. Open WebUI runs with `WEBUI_AUTH=false` for convenience — flip it
  on (`docker compose down webui && docker compose up -d webui` after removing
  the env var) before ever exposing the UI beyond localhost.
