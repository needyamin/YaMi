# Serving: From Dev Server to Production

## Today: local dev API

`fontaine serve` (stdlib-only) exposes `/generate` (JSON + SSE streaming) and
`/health`. Contract:

```
POST /generate   {"prompt": "...", "stream": true, "temperature": 0.7}
→ data: {"delta": "..."} × N      (stream)
→ {"text": "..."}                 (non-stream)
```

One `Generator` per process; requests serialize on the model lock. Good for
local tooling and integration tests, not for exposure beyond localhost.

## Production target architecture

```mermaid
flowchart LR
    C[Client] --> GW[API Gateway<br/>TLS · rate limits]
    GW --> AUTH[Authentication<br/>keys / OAuth]
    AUTH --> Q[Request Queue<br/>backpressure]
    Q --> MS[Model Server<br/>scheduler]
    MS --> W1[GPU Worker × N<br/>batcher + Generator]
    W1 --> M[Model<br/>quantized, KV-cached]
    W1 -->|stream chunks| C
    MS --> REG[Model Registry<br/>which weights]
```

## Evolution steps, in order

1. **Worker-per-process** — keep the stdlib API; run N worker processes with
   one `Generator` each (stateless requests, easiest scale-out, no new deps).
2. **Async gateway + queue** — FastAPI/uvicorn front end (or nginx) enqueuing
   to workers (Redis / SQSR / Celery); streaming stays SSE.
3. **Continuous batching** — a scheduler packs concurrent requests into one
   forward pass; requires batched `KVCache` (already batch-shaped) and
   per-request stop handling in the decode loop.
4. **Quantized GPU workers** — int8/int4 weights, paged KV; capacity ↑ 4–8×.
5. **Horizontal GPU fleet + autoscaling** — workers become stateless pods
   pulling from the queue; the registry decides which checkpoint each loads.
6. **Multi-region / routing** — gateway-level concerns, orthogonal to
   Fontaine internals.

## What Fontaine guarantees at every step

- The request/response contract (`prompt` + sampling params → deltas/text)
  is stable; clients don't change between tiers.
- The `Generator` interface is the *only* thing workers depend on — batching,
  quantization, and sharding are internal replacements.
- Every served response is traceable: the registry maps the deployment to a
  checkpoint, which maps to config + dataset + tokenizer versions.
