# Memory Strategy on 16 GB RAM

## Where training memory goes

AdamW training with fp32 master weights costs, per parameter:

| Component | Bytes/param | Notes |
| --- | --- | --- |
| Parameters | 4 | fp32 weights |
| Gradients | 4 | one per parameter |
| AdamW moments | 8 | `m` and `v`, each fp32 |
| **Subtotal** | **16** | **before a single activation** |

Activations scale with `num_layers × batch × seq × hidden × c` (c ≈ 24 bytes
fp32 in this codebase's estimate; attention adds seq² terms that the estimator
roughs in conservatively). Gradients exist for every parameter *simultaneously*,
which is why training memory is several × model size.

`fontaine model inspect` and `optimization.estimate_training_memory` compute
exactly this breakdown analytically (no GPU required) — cross-checked against
instantiated parameter counts by tests.

## Reference numbers (fp32 + AdamW, activations at batch 8 × seq 512)

| Model | Params | Params+grad+optim | Rough total w/ activations | 16 GB verdict |
| --- | --- | --- | --- | --- |
| Fontaine Tiny | ~3–5 M | ~64 MB | ~0.3–0.5 GB | trivial, CPU ok |
| Fontaine Small | ~40–60 M | ~0.8–1.0 GB | ~2–4 GB | fine |
| Fontaine Medium | ~130–160 M | ~2.5 GB | ~8–14 GB | tight; needs bf16/GPU or aggressive techniques |
| 1 B | — | 16 GB | 25 GB+ | **not trainable on this machine** |
| 70 B | — | 1.1 TB | much more | multi-node FSDP territory |

## What IS feasible on 16 GB RAM

- Fontaine Tiny/Small training (CPU or modest GPU), comfortably.
- Fontaine Medium *architecture* experiments with bf16 + gradient
  checkpointing + small batches on a 16 GB-VRAM GPU.
- Inference of models up to roughly ~1.5–2 B parameters (int8/int4 quantized).
- Data pipelines over corpora far larger than RAM (memmap + streaming).

## What is NOT feasible (be honest)

- Pretraining ≥ 1B-parameter models — memory and compute both fail
  (a 1 B model at Chinchilla data ratios needs ~10²³ FLOPs; CPU/16 GB RAM
  is 5–6 orders of magnitude short in time-to-train).
- Long-context training of large models without GPU memory.
- Serving a 70B-class model locally.

## The 16 GB toolbox (all implemented or seam-ready)

| Technique | Status | Effect |
| --- | --- | --- |
| Streaming datasets + memmap shards | **implemented** | dataset size ∉ RAM |
| Gradient accumulation | **implemented** | effective batch without activation cost |
| Mixed precision (bf16/fp16) | **implemented** | ~½ activation + GPU compute speedup |
| Gradient checkpointing | **implemented** | ~num_layers× fewer stored activations for ~30% compute |
| Configurable seq len / batch / model size | **implemented** | the primary knobs |
| Checkpoint pruning (`keep_last_n`) | **implemented** | bounded disk |
| 8-bit optimizers / CPU offload (bitsandbytes/DeepSpeed) | seam: optimizer factory | ~8× optimizer-state savings |
| Quantization for inference (int8/int4) | seam: `Generator` | 2–8× inference memory savings |
| FSDP / ZeRO sharding | seam: `TrainingStrategy` | scales parameters across GPUs/nodes |

## Practical guidance for the initial machine

1. Start with `configs/model/tiny.yaml`; verify loss falls (`fontaine train`
   + the built-in evaluation).
2. Check the plan first: `fontaine model inspect` prints parameter counts and
   the full memory breakdown for your batch size.
3. When OOM: reduce `batch_size`, raise `gradient_accumulation_steps`
   (keeps effective batch), enable `gradient_checkpointing`, shorten
   `data.sequence_length`.
4. Move to GPU before growing the model; move to bf16 there.
