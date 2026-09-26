# Inference Architecture

Inference is deliberately separate from training: `fontaine.inference` runs
models in eval mode under `torch.inference_mode` and owns the KV cache; the
trainer never imports it and vice versa.

```mermaid
flowchart LR
    P[Prompt] --> T[Tokenizer<br/>encode]
    T --> PF[Prefill<br/>whole prompt, KV cache fill]
    PF --> LOOP[Decode loop<br/>one token/step, O(1) cache]
    S[Sampler<br/>temp · top-k · top-p · rep-penalty] --> LOOP
    LOOP --> ST[Stop conditions<br/>eos · stop sequences · max tokens]
    ST --> OUT[Streamed deltas → text]
```

## The `Generator` (`inference/engine.py`)

- **Prefill** — the whole prompt is processed in one forward pass, filling the
  KV cache; decoding then runs one token per step against cached K/V (O(1)
  attention cost per token instead of O(T)).
- **Sampling** (`generation/sampling.py`, pure functions):
  `temperature` (0 = greedy), `top_k`, `top_p` (nucleus), `repetition_penalty`
  (divide positive / multiply negative logits of generated tokens),
  `max_new_tokens`, `stop_sequences`, `seed` (one seeded generator per
  request → reproducible outputs).
- **Streaming** — `generate(prompt, stream=True)` yields only the newly
  decoded deltas; joining them reproduces the full text (test-enforced).
- **Context management** — `fit_context` keeps the prompt and the requested
  answer when both fit in `max_sequence_length`. When they do not, it keeps
  the tail of the prompt (at most 75% of the window) and spends the rest on
  the answer (at least one token, never more than requested). Generation
  stops before the KV cache overflows.
- **Isolation** — one lock serializes requests sharing a model/cache;
  concurrency = one Generator per worker process.

## KV cache design

`models/kv_cache.py` pre-allocates `[num_layers, batch, num_kv_heads,
max_sequence_length, head_dim]` tensors once and slices per step — avoids
per-step concatenation (quadratic copies) and mirrors production serving
allocators. The cache stores the *compact* shared K/V under GQA. Cache-vs-full
forward equality is unit-tested.

## Loading from a checkpoint

`load_generator(checkpoint_path, tokenizer_dir)` rebuilds the model from the
architecture snapshot stored in the checkpoint's `meta.json`, verifies
integrity, and returns a ready `Generator`.

## Evolution seams

| Capability | How it attaches |
| --- | --- |
| Quantization (int8/int4) | weight-only quantization of the loaded model; `Generator` unchanged |
| Paged / dynamic KV | replace `KVCache` internals |
| Continuous batching | batch dimension of `KVCache` + a scheduler in front of `Generator` |
| Multi-GPU serving | pipeline/tensor-sharded model behind the same `generate` |
| CPU inference | works today (device `auto`); quantization makes it practical |
| Serving APIs | `docs/deployment/serving.md` |

## Dev HTTP API (`inference/server.py`, stdlib-only)

- `GET /health` → `{"status": "ok"}`
- `POST /generate` → `{"prompt": "...", "stream": false, "temperature": 0.5}`
  - `stream: false` → `{"text": ...}`
  - `stream: true` → SSE: `data: {"delta": "..."}` … `data: {"done": true}`

```bash
fontaine serve --checkpoint <ckpt> --tokenizer-dir datasets/tokenizer
curl -X POST localhost:8321/generate -d '{"prompt": "Hi", "stream": false}'
```

The same process also speaks the Ollama-compatible API (`/api/chat`,
`/api/tags`, `/api/ps`, `/api/show`) so Open WebUI can attach with no model
conversion. Route details, the Alpaca chat template, and active versus total
parameter reporting are in `docs/deployment/serving.md`. This server is a
development convenience — the production serving path is a different tier.
