Connect Fontaine to a ready-made open-source UI (Open WebUI) — no UI development

Goal: replace the custom HTML playground with the existing, popular open-source chat UI (Open WebUI). Your model needs no changes and no conversion — the only work is a small backend API bridge so `fontaine serve` speaks the standard Ollama API language that Open WebUI (and any Ollama-compatible UI) already speaks.

Code changes:
1. New `src/fontaine/inference/chat_template.py` — joins `{system, user, assistant}` messages into one prompt (Alpaca-style template, matching your CodeAlpaca training data). Small and unit-testable.
2. Ollama-API endpoints in `src/fontaine/inference/server.py` (stays stdlib-only, matching the existing code style):
   - `GET /api/version` — UIs probe this first
   - `GET /api/tags` — model list for the UI's model picker (name from the checkpoint run folder)
   - `POST /api/chat` — receives the full chat history `{messages: [...]}` (stateless, like real Ollama), streams NDJSON `{"message":{"content":delta},"done":false}` → `{"done":true}`; maps `options` (temperature, top_k, top_p, num_predict, stop) to the engine's sampling config
   - `POST /api/generate` — raw-prompt variant (`{"response": delta, ...}`)
3. `GET /` now returns a small status JSON instead of the custom page. `POST /generate` and `GET /health` stay (internal API for curl/tests).
4. Delete `src/fontaine/inference/webui.py` — the custom playground, 213 lines.

Docker:
5. `docker-compose.yml`: `web` service stays; add `webui` service running `ghcr.io/open-webui/open-webui` with `OLLAMA_BASE_URL=http://web:8321`, exposed on port 3000. You open http://localhost:3000 → your model is in the picker → chat.
6. `.env.example`: add the WebUI port variable.

Tests + docs:
7. New `tests/test_ollama_api.py`: endpoint round-trips with the existing tiny test fixtures (tags, streaming chat shape, generate); unit test for the chat template.
8. Update `docs/deployment/serving.md`, `docs/deployment/docker.md`, and the README Docker section to document Open WebUI instead of the custom playground.

Out of scope: GGUF export for the real Ollama runtime — unnecessary for this and can be a separate tool later if you ever want `ollama run fontaine`.

Verification: pytest green; curl checks that `/api/chat` streaming output matches Ollama's documented NDJSON shape; `docker compose up` end-to-end — Open WebUI lists the model and streams a chat.