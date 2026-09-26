# Scaling Roadmap

## Size ladder

Architectural targets — not promises that any size trains on the initial
16 GB machine.

| Stage | Params (≈) | hidden × layers | Context | Train state before activations | Hardware tier |
| --- | --- | --- | --- | --- | --- |
| Fontaine Tiny | ~5.2 M dense | 256 × 4 | 256 | ~80 MB | CPU / any GPU — **today** |
| Fontaine Small | ~28 M dense | 512 × 8 | 512 | ~0.4 GB | modest GPU |
| Fontaine Medium | ~82 M dense | 768 × 12 | 1024 | ~1.2 GB | free GPU (Kaggle T4) |
| Coding Low | 5.7 M active / 22 M total | 256 × 6, 8 experts top-1 | 512 | ~0.3 GB | laptop CPU, 16 GB RAM |
| Coding Mid | 39 M active / 114 M total | 512 × 8, 8 experts top-2 | 1024 | ~1.8 GB | desktop CPU or free GPU |
| Coding High | 129 M active / 384 M total | 768 × 12, 8 experts top-2 | 2048 | ~6 GB | GPU to train; workstation CPU to run |
| Fontaine Large | 400–700 M | 1024 × 24 | 2048 | 40–80 GB | single A100/H100 class |
| Fontaine XL | 1–4 B | 2048 × 24–32 | 4096 | multi-GPU | 4–8 GPUs, FSDP/ZeRO |
| Distributed Fontaine | 8–70 B+ | 4096–8192 × 32–80 | 8k–128k | cluster | multi-node FSDP + TP/PP |

Dense rows assume an 8k vocabulary. Coding rows are `moe_decoder`: total
parameters count every expert (that is what AdamW stores); active parameters
are what one token multiplies. `fontaine model inspect` prints both.

Context lengths assume RoPE theta scaling plus training-data length mix.

## Scaling dimensions and what they hit

| Dimension | RAM (host) | VRAM (GPU) | Disk | Training time | Inference latency |
| --- | --- | --- | --- | --- | --- |
| Parameters | checkpoints, data prep | weights+grads+optim+acts | checkpoints (∝) | ∝ params | ∝ params (weights touched per token) |
| Context length | — | attention + KV cache (GQA dents it) | — | ∝ ~seq² attention | KV cache size ∝ seq |
| Vocabulary | shard dtype | embedding + head | — | mildly | logits + KV head dim |
| Data size | pipeline is streaming | — | shards ∝ data | ∝ tokens (Chinchilla ≈ 20 tokens/param) | — |
| Compute (steps) | — | — | more checkpoints | ∝ steps | — |
| Batch size | — | activations | — | fewer, larger steps | — |
| Multi-GPU | — | per-GPU share ↓ | — | ↓ (communication overhead) | ↓ with sharding |
| Multi-node | — | — | object storage | ↓ (network-bound) | serving capacity ↑ |

Chinchilla-style guidance: at fixed compute, parameters and training tokens
should scale together (~20 tokens/param). Data is usually the cheaper
resource — scale the model only when the data pipeline feeds it.

## Parallelism path (as models outgrow single GPUs)

```mermaid
flowchart LR
    S1[Single device<br/>today] --> S2[DDP<br/>replicate model,<br/>shard data]
    S2 --> S3[FSDP / DeepSpeed ZeRO<br/>shard params+optim+grads]
    S3 --> S4["Tensor parallel<br/>shard matrices within layers<br/>(Megatron-style)"]
    S4 --> S5[Pipeline parallel<br/>shard layers across devices]
    S5 --> S6[3D / multi-node<br/>TP × PP × DP]
```

| Strategy | Use when | Cost | Fontaine seam |
| --- | --- | --- | --- |
| Gradient accumulation | OOM on batch | none | implemented |
| Gradient checkpointing | OOM on activations | ~30% recompute | implemented |
| DDP | model fits one GPU | gradient all-reduce | `DistributedDataParallelStrategy` (skeleton) |
| FSDP/ZeRO | model > one GPU | comms + prefetching | `TrainingStrategy` |
| Tensor parallel | layers > one GPU | all-reduce per layer, NVLink wanted | `build_model`/attention internals |
| Pipeline parallel | depth-bound clusters | bubble/stashing | block list is already layered |
| Distributed checkpoints | any sharded run | — | v1 format records `world_size` |
| Distributed data loading | many workers | — | `DistributedSampler` already wired |

## What changes at each tier — and what doesn't

- **Configs** grow new YAML files; the schema never mutates meaning.
- **`TrainingStrategy`** gains implementations; the `Trainer` loop stands.
- **Checkpointing** shards per rank behind the same manager API.
- **Data** moves to object storage + distributed preprocessing behind the same
  manifest contract.
- **Inference** moves to batched GPU workers behind the same `Generator`
  contract.

Nothing above requires *rewriting* Fontaine — each is an implementation
behind an interface that already exists.
