# Scaling Roadmap

## Size ladder

Architectural targets — not promises that any size trains on the initial
16 GB machine.

| Stage | Params (≈) | hidden × layers | Context | VRAM to *train* (bf16, est.) | Hardware tier |
| --- | --- | --- | --- | --- | --- |
| Fontaine Tiny | 3–5 M | 256 × 4 | 256 | < 1 GB | CPU / any GPU — **today** |
| Fontaine Small | 40–60 M | 512 × 8 | 512 | 4–8 GB | modest GPU |
| Fontaine Medium | 130–160 M | 768 × 12 | 1024 | 16–32 GB | 16 GB+ GPU + offload/ckpt |
| Fontaine Large | 400–700 M | 1024 × 24 | 2048 | 40–80 GB | single A100/H100 class |
| Fontaine XL | 1–4 B | 2048 × 24–32 | 4096 | multi-GPU | 4–8 GPUs, FSDP/ZeRO |
| Distributed Fontaine | 8–70 B+ | 4096–8192 × 32–80 | 8k–128k | cluster | multi-node FSDP + TP/PP |

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
