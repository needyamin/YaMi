# YaMi super profiles

`configs/yami_super.yaml` is one configuration file with a profile ladder.
`python -m fontaine.cli.main model estimate-params --config configs/yami_super.yaml --profile <name>`
counts parameters and memory without allocating the model. The same command
with `model feasibility` and `distributed plan` checks a machine and a process
count. Neither command starts training.

Profile names are architectural targets. The numbers below are the estimator
output at vocabulary 32,000 with tied embeddings, so the LM head adds no extra
parameters. A trillion-parameter configuration is an engineering plan. It is
not a trained model and it is not intelligent.

## What each example is

| Profile | Status | Why |
| --- | --- | --- |
| `tiny` | Tested | Constructed, compared with `num_parameters()`, and benchmarked (2 train steps). |
| `small`, `medium`, `500m` | Theoretically supported | Config validates and the estimator runs. The module is not allocated in tests. |
| `1b`, `3b`, `7b`, `14b` | Hardware-dependent | Same estimator path. Training memory is tens to hundreds of GiB. |
| `30b`, `70b`, `100b`, `300b`, `1t` | Hardware-dependent | Mixture-of-experts. Planned with `distributed plan`. Not instantiated. |

## Counts

Dense profiles use every parameter on every token.

| Profile | Total parameters | Context | Training memory (this file's micro-batch) |
| --- | --- | --- | --- |
| tiny | 10,554,112 | 512 | 173.04 MiB |
| small | 41,559,552 | 1024 | 658.15 MiB |
| medium | 100,094,208 | 2048 | 1.53 GiB |
| 500m | 453,905,664 | 4096 | 6.82 GiB |
| 1b | 1,057,586,688 | 4096 | 15.85 GiB |
| 3b | 2,828,978,176 | 4096 | 42.30 GiB |
| 7b | 5,802,045,440 | 8192 | 86.64 GiB |
| 14b | 13,376,406,528 | 8192 | 199.56 GiB |

Mixture-of-experts profiles store every expert. Active parameters are the ones
multiplied for one token (embeddings, attention, norms, router, shared experts,
and the top-k routed experts).

| Profile | Total | Active per token | Training memory |
| --- | --- | --- | --- |
| 30b | 28,856,037,376 | 6,307,459,072 | 430.18 GiB |
| 70b | 75,064,261,632 | 13,055,671,296 | 1.09 TiB |
| 100b | 104,995,567,616 | 14,398,601,216 | 1.53 TiB |
| 300b | 295,694,647,296 | 34,775,384,064 | 4.30 TiB |
| 1t | 1,244,665,499,648 | 46,369,624,064 | 18.11 TiB |

Training memory is fp32 weights, gradients, AdamW moments, and a rough
activation term for the batch and sequence in that profile. The KV-cache line
in the estimator is inference-only and is not added to the training total.
The 1t figure is the analytic cost of that micro-batch, not a claim that one
server can hold it. `training.fit_batch_to_memory: true` lowers the micro-batch
until the estimate fits measured RAM or GPU memory, and raises if even one
sequence does not fit.

## Architecture

Decoder-only Transformer: RMSNorm, RoPE (`none`, `linear`, `ntk`), optional
QK-norm, SwiGLU or GELU, and tied embeddings. `head_dim: 0` means
`hidden / heads`. A positive even `head_dim` is the projection width.

Attention backends:

- `auto` and `optimized` use scaled dot-product attention.
- `reference` uses explicit matmul and softmax.
- `flash` requires CUDA flash attention and raises on CPU or when a mask is
  required. It does not fall back.

`layer_pattern` tiles with modulo (`full`, `sliding`, `sparse`). Sparse
patterns are `local`, `block`, `strided`, `global-local`, and `hybrid`.
`hybrid` requires a layer pattern. Optional `document_ids` mask attention
inside packed documents and cannot be combined with a KV cache.

Kernels are `auto`, `optimized`, and `reference` for RMSNorm, SwiGLU, and MoE
dispatch. `optimized` RMSNorm requires `F.rms_norm` and raises if it is missing.
`reference` MoE walks tokens one by one when capacity is off and expert
parallel is 1.

## Mixture of experts

Token-choice router, no bias, top-k. The Switch auxiliary loss is added only
while training, and only when `load_balancing_enabled` is true. Shared experts
are real feed-forward modules, always on, and added after the expert
all-reduce so expert parallel does not multiply them. `expert_capacity_factor: 0`
keeps every selected token. A positive factor keeps the highest router weights
per expert and records dropped tokens. `routing_stats()` reports utilization,
dropped tokens, entropy, and load imbalance.

## Distributed training

`world_size = data_parallel * tensor_parallel * pipeline_parallel`. Expert
parallel must divide data parallel and `num_experts`. Sequence parallel is 1
or equal to tensor parallel, and the sequence length must divide by it. Tensor
parallel must divide heads, KV heads, and the feed-forward width. Pipeline
parallel cannot exceed the layer count. A mismatch with the process group
raises and tells you to launch with `torchrun`. The planner does not start
training.

Rank layout is data, then pipeline, then tensor (tensor ranks are adjacent).

- Data parallel averages gradients on the data-parallel group.
- Tensor parallel shards Q, K, V, O and the feed-forward linears. Column
  outputs are gathered. Row outputs are summed. The collectives are in the
  autograd graph.
- Pipeline parallel builds only that stage's layers. The forward sends
  activations; the backward sends gradients. One micro-batch is in flight.
  Tied embeddings across stages are rejected. A KV cache with pipeline
  parallel is rejected.
- Sequence parallel splits the feed-forward sequence across the tensor-parallel
  ranks and all-gathers the result. Attention still sees the full sequence.
- Expert parallel keeps a slice of the experts and all-reduces their outputs.
- `training.sharding: fsdp` wraps FullyShardedDataParallel.
  `optimizer` uses ZeroRedundancyOptimizer. Both require a process group.
- `training.precision` supports fp32, fp16, and bf16 for the training autocast.

`distributed plan --world-size N` prints a mesh. Example that fits the rules:
data 8, tensor 4, pipeline 2, expert 4, world 64. Expert 4 does not divide
data 2, so that combination is rejected with the reason.

## Data

Streaming token shards are unchanged when the new flags are off. A `mixture`
list samples sources by weight. `pack_documents` writes a parallel document-id
shard. Quality filters run only when their flags are set: banded SimHash
near-duplicate removal, regex PII redaction, malformed-line drops, a Unicode
script check, and exact-hash plus 5-word shingle contamination. There is no
learned quality score.

## Training stages

`pretraining` is the original next-token loop. `supervised_fine_tuning` masks
labels before `### Response:\n` and raises if a batch has no supervised
tokens. `preference_training` reads JSONL and applies DPO with beta 0.1.
`reinforcement_learning` scores the dataset answer with REINFORCE. It does
not sample a rollout.

Logs include loss, learning rate, gradient norm, tokens per step, tokens per
second, samples per second, and, when the model has experts, utilization and
routing entropy. CUDA runs also record allocated bytes. Each experiment
directory stores the resolved config plus git revision, tokenizer, dataset
path, seed, precision, optimizer, scheduler, topology, and a hardware snapshot.

## Checkpoints

A single-process checkpoint is still `model.pt`, `optimizer.pt`, `rng.pt`, and
`meta.json` with SHA-256 hashes. `training.async_checkpoints: true` copies
tensors to CPU and writes that checkpoint on a background thread. The next
save and the end of `fit` wait for it. Async writing is rejected when the
world size is above 1, because the latest pointer cannot be published until
every rank has finished its shard.

When tensor, pipeline, or expert parallel is above 1, each rank writes
`model_rank{r}.pt` (and matching optimizer and RNG files). Hashes are gathered
into one `meta.json`. Loading requires the same process count. Shards are not
gathered onto one device.

## Inference

Generation supports greedy, temperature, top-k, top-p, repetition penalty, stop
sequences, and a token cap. KV caches are `static` (preallocated), `dynamic`
(grows), and `paged` (page size 16). Streaming is the existing generator.
`inference.batching` holds a continuous batcher with one dynamic cache per
request.

Speculative decoding runs only at temperature 0. `draft_checkpoint` is loaded
with the target. A non-greedy speculative request raises.

Tensor-parallel inference slices a full checkpoint after loading. Pipeline and
expert parallel inference require a checkpoint saved with that mesh. A full
checkpoint is not silently cut into pipeline stages.

## Quantization

Inference precisions that execute: fp32, fp16, bf16, int8, int4. `auto` still
never selects int4. int8 is CPU-only. int4 uses a group size of at most 32.
Embeddings and norms stay fp32. A tied LM head gets its own quantized copy.
Quantized modules run a forward in the precision tests.

## Evaluation, post-training, agents

Evaluators: validation loss, exact match, math, and a needle-in-a-haystack
prompt. Verifiers: exact match, mathematical comparison, a Python snippet, and
a unit-test check. Custom verifiers register by name.

The agent loop has a filesystem tool and a Python tool inside a directory
jail. On Windows, a request for memory or CPU rlimits raises; those limits are
not pretended. Network deny is an allowlist of command names, not a network
namespace.

## Modalities

`model.modalities` must be `["text"]`. Any other entry fails validation. There
is no image, audio, or video encoder.

## Hardware

`hardware inspect` reports CPU, RAM, available RAM, CUDA devices, an
interconnect string from `nvidia-smi topo -m` when that command works, and the
precisions and attention backends this process can use. Missing data stays
missing. Feasibility does not allocate weights. It compares the training and
inference estimates with the machine. Full fine-tuning is the same AdamW
estimate as pretraining. LoRA and other parameter-efficient methods are not
implemented.

## Benchmark

`benchmark --profile tiny --allow-large` on this development machine allocated
the 10,554,112-parameter profile and ran 2 forward/backward steps (batch 2,
sequence 32) in 0.041 seconds, about 3,136 tokens/second. Profiles above 5
million parameters are refused unless `--allow-large` is set, so a large
config is not allocated by accident.

## Limits that are still real

- Context has no hard-coded ceiling. Profiles stop at 8,192. Tests use short
  sequences. Longer windows are a config change plus memory.
- The largest module the tests and the tiny benchmark construct is the 10.6M
  profile. 1B through 1T were estimated and planned only.
- Pipeline parallel moves one micro-batch. It does not schedule a pipeline
  bubble fill.
- Preference training uses beta 0.1. Reinforcement learning does not sample.
- Flash attention, multi-node runs, and GPU utilization depend on the machine.
- A large parameter count does not create model quality. Quality still depends
  on data, compute, optimization, post-training, evaluation, and inference.
