# Development Roadmap & Definition of Done

Each phase has objective, verifiable exit criteria. Phases 0–6 are
**complete** in this repository.

## Phase 0 — Architecture ✅
Deliverables: docs, repo structure, config schemas, interfaces.
**Done when:** every subsystem has a package, an interface, and a doc page;
configs drive behavior without source edits; `pytest` green.

## Phase 1 — Minimal Transformer ✅
**Done when:** decoder-only model (RoPE, GQA, SwiGLU, RMSNorm) forward/backward
passes; causality and KV-cache≡full-forward tests pass; parameter count
verified analytically; `fontaine model inspect` reports sizes + memory.

## Phase 2 — Tokenizer + Data Pipeline ✅
**Done when:** tokenizer trains/saves/loads from streaming corpus (char +
byte-level BPE); raw txt/jsonl/csv/json → cleaned, deduplicated, packed,
checksummed shards + manifest; dataset trains from memmap without loading
corpus into RAM; manifest validation catches tampering.

## Phase 3 — Training ✅
**Done when:** `fontaine train` runs end-to-end (config-driven) with
accumulation, warmup+cosine LR, clipping, AMP, logging, evaluation hooks;
overfit-the-tiny-dataset sanity test passes; CLI integration test passes.

## Phase 4 — Checkpointing ✅
**Done when:** atomic, hash-sealed checkpoints with complete state (model,
optimizer, scheduler, scaler, RNG, metadata); resume continues step count and
metrics history; best/latest pointers + pruning verified by tests.

## Phase 5 — Evaluation ✅
**Done when:** registry-based evaluators; validation loss + perplexity
implemented; new evaluator addable by registration + config line without
trainer changes (test-enforced); `fontaine evaluate` works standalone.

## Phase 6 — Inference ✅
**Done when:** KV-cached generation with temperature/top-k/top-p/repetition
penalty/stop sequences/max tokens; streaming deltas; deterministic seeding;
`fontaine generate` + dev `/generate` API; generation correctness tests pass.
The same server now also speaks the Ollama-compatible chat API used by
Open WebUI (`/api/chat`, `/api/tags`, `/api/ps`); see
`docs/deployment/serving.md`.

## Phase 7 — Fine-Tuning (next)
**Done when:** instruction-format dataset adapter (prompt-masked labels);
SFT configs; SFT checkpoint trains from a pretrained one; instruction-following
evaluator registered.

## Phase 8 — Multi-GPU
**Done when:** `DistributedDataParallelStrategy` validated on a real multi-GPU
box via torchrun; `DistributedSampler` path exercised; metrics sync and
primary-rank-only artifact writes proven; per-device memory profiled.

## Phase 9 — Distributed Training
**Done when:** FSDP (or DeepSpeed ZeRO) strategy behind `TrainingStrategy`
trains a model that does not fit one GPU; sharded checkpoints round-trip;
multi-node launch documented; manifest-based distributed data loading.

## Phase 10 — Large-Scale Infrastructure
**Done when:** object-storage data/checkpoint backends; model registry service
with promotion gates; production serving (queue + continuous batching +
quantized workers); post-training pipeline (SFT → DPO) operational.

## Tracking

| Phase | Status | Evidence |
| --- | --- | --- |
| 0–6 | ✅ complete | 116 passing tests + CLI integration test in this repo |
| 7–10 | designed, seams in place | docs: `research/`, `scaling/`, `deployment/` |
