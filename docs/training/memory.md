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

## Reference numbers

Parameter counts are what `fontaine model inspect` prints at an 8k vocabulary.
Training state is 16 bytes per **total** parameter (fp32 weights + gradients +
AdamW). Activations use this file's estimator at batch 8 and that model's own
context (`gradient_checkpointing` off). Coding rows are mixture-of-experts:
AdamW pays for every expert, so the state column uses the total, not the
active count.

| Model | Params | Params+grad+optim | Rough total w/ activations | 16 GB verdict |
| --- | --- | --- | --- | --- |
| Fontaine Tiny | ~5.2 M dense | ~80 MB | ~0.13 GB (seq 256) | trivial, CPU ok |
| Fontaine Small | ~28 M dense | ~0.4 GB | ~0.8 GB (seq 512) | fits; a GPU is much faster |
| Fontaine Medium | ~82 M dense | ~1.2 GB | ~2.9 GB (seq 1024) | fits in RAM; train on a GPU |
| Coding Low | 5.7 M active / 22 M total | ~0.3 GB | ~0.5 GB (seq 512) | laptop CPU, 16 GB RAM |
| Coding Mid | 39 M active / 114 M total | ~1.7 GB | ~2.5 GB (seq 1024) | desktop CPU or a free GPU |
| Coding High | 129 M active / 384 M total | ~5.7 GB | ~9 GB (seq 2048) | workstation to run; GPU to train |
| 1 B | — | 16 GB | 25 GB+ | **not trainable on this machine** |
| 70 B | — | 1.1 TB | much more | multi-node FSDP territory |

## What IS feasible on 16 GB RAM

- Fontaine Tiny and Coding Low training on CPU, comfortably.
- Fontaine Small and Coding Mid in 16 GB RAM; a modest or free GPU is the
  practical trainer.
- Fontaine Medium (~82 M, about 3 GB at batch 8 × 1024 in fp32). It fits in
  16 GB RAM; a GPU is still how you actually train it. Coding High needs a
  GPU and gradient checkpointing.
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
