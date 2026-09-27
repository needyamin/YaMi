# MASTER ENGINEERING PROMPT — YaMi FRONTIER-SCALE LLM

You are the principal AI systems architect and ML infrastructure engineer working directly inside the existing YaMi repository.

Your mission is to transform YaMi into a **general-purpose, frontier-scale large language model framework** capable of scaling from millions of parameters to billions and eventually trillions of parameters, limited primarily by available hardware, distributed infrastructure, training data, and compute.

DO NOT optimize YaMi for a single model size.

Build the architecture so that the same system can evolve continuously:

```
Million
  ↓
10M
  ↓
50M
  ↓
100M
  ↓
500M
  ↓
1B
  ↓
3B
  ↓
7B
  ↓
14B
  ↓
30B
  ↓
70B
  ↓
100B+
  ↓
1T+
  ↓
multi-trillion parameters
```

The framework must scale according to available hardware instead of imposing artificial architectural limits.

---

# ABSOLUTE RULES

1. Inspect the entire existing YaMi repository before changing anything.
2. Understand every existing subsystem before modifying it.
3. Preserve working functionality unless there is a strong technical reason to replace it.
4. Reuse existing YaMi abstractions whenever possible.
5. Do not create duplicate implementations.
6. Do not create fake implementations.
7. Do not create placeholder functions that pretend to work.
8. Do not add configuration fields that are not implemented.
9. Every YAML option must correspond to real executable behavior.
10. Unsupported features must fail clearly instead of silently falling back.
11. Do not hard-code model dimensions.
12. Do not hard-code a maximum parameter count.
13. Do not hard-code a maximum context length unless imposed by hardware/runtime limitations.
14. Do not hard-code a maximum number of experts.
15. Do not hard-code a maximum number of GPUs/nodes.
16. Avoid architecture-specific assumptions wherever possible.
17. Keep components modular.
18. Every major feature must have tests.
19. Benchmark performance after significant infrastructure changes.
20. Do not claim frontier-level intelligence merely because a large configuration can be instantiated.
21. Clearly separate:

    * architecture scalability
    * training scalability
    * inference scalability
    * data scalability
    * actual model capability.

---

# PRIMARY OBJECTIVE

Create a new scalable configuration:

```
configs/yami_super.yaml
```

This becomes the main configurable architecture for YaMi.

It must allow the same implementation to construct models ranging from tiny experimental models to trillion-parameter distributed MoE architectures.

The model size must be DERIVED from architecture.

Never use:

```
parameter_count: 1000000000000
```

as the mechanism for constructing the model.

Instead calculate parameter count from:

```
vocabulary
hidden dimensions
layers
attention dimensions
FFN dimensions
number of experts
expert dimensions
shared experts
embeddings
output projection
normalization
other trainable components
```

---

# PARAMETER SCALING ENGINE

Implement a robust parameter calculator.

Command:

```
yami model estimate-params --config configs/yami_super.yaml
```

Output:

```
Embedding parameters
Attention parameters
Q parameters
K parameters
V parameters
O projection parameters
Dense FFN parameters
MoE expert parameters
Router parameters
Shared expert parameters
Normalization parameters
LM head parameters
Total parameters
Active parameters per token
Estimated optimizer memory
Estimated parameter memory
Estimated activation memory
Estimated KV-cache memory
Approximate total training memory
```

For MoE:

```
total_parameters != active_parameters
```

Always report both.

The estimator must work WITHOUT allocating the complete model.

A trillion-parameter configuration must be inspectable on a normal development machine.

---

# HARDWARE-AWARE SCALING

YaMi must understand hardware constraints.

Create a hardware inspection system:

```
yami hardware inspect
```

Report:

```
CPU
RAM
GPU count
GPU model
VRAM
CUDA availability
accelerator backend
interconnect where detectable
available system memory
supported precision
available attention kernels
```

Create:

```
yami model feasibility --config ...
```

The command should estimate whether the selected configuration is feasible for:

```
inference
training
fine-tuning
```

and report estimated:

```
GPU memory
system memory
number of accelerators
parallelism requirements
estimated checkpoint size
```

Never automatically allocate an enormous model simply to inspect it.

---

# MODEL ARCHITECTURE

Build a modern decoder-only Transformer architecture.

Required architectural components:

```
RMSNorm
RoPE
GQA
SwiGLU
QK-Norm
KV Cache
causal attention
configurable attention backend
```

Architecture must support:

```
dense Transformer
MoE Transformer
hybrid dense/MoE Transformer
```

Every architectural component must be independently configurable.

---

# MODULAR MODEL SYSTEM

Create modular interfaces:

```
Model
TransformerBlock
Attention
AttentionBackend
FeedForward
MoELayer
Router
Expert
Normalization
PositionalEncoding
KVCache
Embedding
LMHead
```

Do not put the complete architecture into one monolithic class.

---

# ATTENTION SYSTEM

Implement an extensible attention backend.

Conceptual structure:

```
AttentionBackend
    ├── Standard
    ├── GQA
    ├── SlidingWindow
    ├── Sparse
    ├── Hybrid
    └── OptimizedKernel
```

Use existing YaMi naming conventions where appropriate.

The backend must be selected through configuration.

Example:

```
attention:
  type: gqa
  backend: auto
```

Supported backends must be detected at runtime.

Explicitly requested unavailable backends must produce an actionable error.

---

# GQA

Support:

```
num_attention_heads
num_kv_heads
head_dim
```

Validate:

```
num_attention_heads % num_kv_heads == 0
```

Support different head dimensions where technically valid.

Avoid assumptions that:

```
hidden_size == num_heads * fixed_head_dimension
```

unless required by the selected architecture.

---

# ROPE

Create a flexible positional encoding subsystem.

Configuration must support:

```
theta
scaling
scaling_type
original_context_length
```

Design it to support progressively larger contexts.

Target infrastructure:

```
4K
8K
16K
32K
64K
128K
256K
512K
1M+
```

Do not claim long-context capability simply because a large context number appears in YAML.

Long-context reliability requires training and evaluation.

---

# LONG-CONTEXT SYSTEM

Implement infrastructure for very long sequences:

```
RoPE scaling
sliding-window attention
sparse attention
hybrid attention
efficient KV cache
KV cache compression hooks
memory-aware batching
sequence packing
chunked processing
```

Do not force all layers to use full quadratic attention.

Architecture must permit layer-level attention patterns.

Example:

```
layer_pattern:
  - full
  - sliding
  - sliding
  - sparse
```

The pattern must be configurable.

---

# SPARSE / HYBRID ATTENTION

Create a generic sparse attention interface.

Support patterns such as:

```
local
block
strided
global-local
hybrid
```

Do not implement arbitrary complexity without measurable benefit.

Benchmark each implementation.

The system must allow future attention algorithms to be plugged in without rewriting the Transformer.

---

# MOE ENGINE

Build a production-grade Mixture-of-Experts system.

Support:

```
num_experts
experts_per_token
expert_intermediate_size
router
shared experts
expert capacity
load balancing
expert parallelism
```

The number of experts must NOT be hard-coded.

Potential configurations must range from:

```
2
4
8
16
32
64
128
256
512
1024+
```

subject to hardware and implementation constraints.

---

# ROUTER

Implement configurable routing.

At minimum:

```
top-k routing
```

Support:

```
router logits
routing probabilities
expert selection
capacity management
overflow handling
load balancing
```

Expose routing statistics:

```
expert utilization
token distribution
dropped tokens
routing entropy
load imbalance
```

---

# LOAD BALANCING

Implement configurable expert load balancing.

Configuration:

```
router:
  load_balancing:
    enabled: true
    coefficient: ...
```

Do not hard-code the coefficient.

Design the routing system so future routing algorithms can be introduced cleanly.

---

# SHARED EXPERTS

Support:

```
zero or more shared experts
```

Shared experts must be implemented as actual trainable components.

Parameter estimation must include them.

---

# EXPERT PARALLELISM

MoE must support distributing experts across accelerators.

Architecture:

```
global router
    ↓
expert dispatch
    ↓
remote/local experts
    ↓
gather
    ↓
residual stream
```

Minimize communication overhead.

Expose:

```
expert_parallel_size
```

and validate it against available devices.

---

# DISTRIBUTED TRAINING

This is a core requirement.

Implement modular distributed-training infrastructure supporting:

```
Data Parallelism
Tensor Parallelism
Pipeline Parallelism
Sequence Parallelism
Expert Parallelism
```

Do not create fake flags.

If a parallelism mode is configured, it must actually affect execution.

---

# PARALLELISM COMPOSITION

Allow combinations such as:

```
data × tensor × pipeline
```

and:

```
data × tensor × pipeline × expert
```

Validate:

```
world_size
tensor_parallel_size
pipeline_parallel_size
expert_parallel_size
```

against the actual distributed environment.

Provide a clear explanation when the requested topology is impossible.

---

# SHARDED TRAINING

Support memory-efficient distributed training through the available framework mechanisms.

Support:

```
parameter sharding
gradient sharding
optimizer-state sharding
```

Avoid unnecessary replication of huge model states.

---

# MIXED PRECISION

Support:

```
FP32
FP16
BF16
```

Architect the precision subsystem so future lower-precision formats can be integrated cleanly.

Do not claim support for a precision format without testing actual forward/backward/inference execution.

---

# OPTIMIZED KERNELS

Create kernel abstraction layers for:

```
attention
normalization
matrix multiplication
MoE dispatch
activation
```

Use optimized implementations when available.

Fallback implementations must remain correct.

Provide:

```
auto
optimized
reference
```

modes where practical.

Reference mode should prioritize correctness and debugging.

---

# MEMORY MANAGEMENT

Implement:

```
activation checkpointing
gradient accumulation
gradient clipping
memory-aware batch sizing
sequence packing
efficient KV cache
asynchronous data loading
pinned memory where appropriate
prefetching
```

Training must not unnecessarily materialize duplicate tensors.

---

# OPTIMIZER SYSTEM

Create an optimizer abstraction.

At minimum support:

```
AdamW
```

Architecture should permit additional optimizers.

Configuration:

```
optimizer:
  type: adamw
  learning_rate: ...
  betas: [...]
  eps: ...
  weight_decay: ...
```

Do not bury optimizer parameters in source code.

---

# LR SCHEDULER

Support:

```
cosine
linear
constant
```

Configuration:

```
scheduler:
  type: cosine
  warmup_steps: ...
  min_lr_ratio: ...
```

---

# TRAINING STAGES

Separate training stages.

Support architecture:

```
pretraining
supervised_fine_tuning
preference_training
reinforcement_learning
```

Do not combine all training logic into one loop.

Each stage should have its own configuration and trainer interface.

---

# LARGE-SCALE DATA PIPELINE

Create a scalable dataset system supporting:

```
local datasets
sharded datasets
streaming datasets
remote/object storage where supported
deterministic sampling
distributed sharding
sequence packing
document packing
dataset mixtures
```

Avoid loading the entire training corpus into RAM.

---

# DATA MIXTURE

Support configurable mixtures:

```
datasets:
  mixture:
    - name: general
      weight: ...
    - name: code
      weight: ...
    - name: mathematics
      weight: ...
    - name: reasoning
      weight: ...
    - name: multilingual
      weight: ...
```

Weights must be configurable.

---

# DATA QUALITY

Build modular data-processing interfaces:

```
language detection
quality scoring
deduplication
near-deduplication
contamination detection
document filtering
PII filtering
malformed-data filtering
```

Do not create fake quality scores.

If a filter is experimental, clearly label it.

---

# TOKENIZER

Maintain a tokenizer abstraction.

Support configurable:

```
tokenizer type
tokenizer path
vocabulary
special tokens
```

Validate:

```
tokenizer vocab size
model vocab size
```

before training.

---

# SEQUENCE PACKING

Implement efficient token packing.

Avoid wasting compute on padding.

Support:

```
packed sequences
attention masks
document boundaries
```

Benchmark packing efficiency.

---

# CHECKPOINT SYSTEM

Checkpoints must contain:

```
model weights
optimizer state
scheduler state
training step
RNG states
tokenizer metadata
configuration
dataset state
distributed state where necessary
```

Support:

```
save
load
resume
validation
compatibility checks
```

---

# LARGE CHECKPOINTS

Design checkpointing for extremely large models.

Support sharded checkpoints.

Never require a trillion-parameter model to be gathered onto one device simply to save it.

Support:

```
asynchronous checkpoint writing
resumable checkpoints
shard metadata
integrity validation
```

---

# INFERENCE ENGINE

Create an inference engine optimized separately from training.

Support:

```
KV cache
streaming
batching
continuous batching architecture
speculative decoding hooks
tensor parallel inference
pipeline parallel inference
quantized inference
```

---

# GENERATION

Support:

```
greedy
temperature
top-k
top-p
repetition penalty
stop sequences
maximum output tokens
```

All generation parameters must be configurable.

---

# KV CACHE

Build a proper KV-cache abstraction.

Support:

```
dynamic cache
static cache
paged-cache architecture
cache reuse hooks
memory accounting
```

Avoid allocating maximum context memory unnecessarily.

---

# QUANTIZATION

Create a quantization subsystem.

Architecture should support:

```
FP16
BF16
INT8
INT4
```

only when actually implemented and tested.

Quantized checkpoints must be loadable and executable.

---

# MODEL PARALLEL INFERENCE

Large models must be able to run across multiple devices.

Support:

```
tensor parallel inference
pipeline parallel inference
expert parallel inference
```

Avoid copying unnecessary parameters between devices.

---

# TRAINING MONITORING

Track:

```
loss
validation loss
learning rate
gradient norm
throughput
tokens/sec
samples/sec
GPU utilization
memory usage
expert utilization
routing statistics
checkpoint time
```

Store experiment metadata.

---

# EXPERIMENT REPRODUCIBILITY

Every training run must record:

```
complete configuration
git revision
tokenizer version
dataset version
random seeds
precision
hardware
distributed topology
optimizer
scheduler
```

A training run should be reproducible as far as the underlying hardware/runtime permits.

---

# EVALUATION SYSTEM

Build a modular evaluation framework.

Support categories:

```
knowledge
reasoning
mathematics
coding
instruction following
long-context retrieval
multilingual
tool use
agent tasks
```

Do not hard-code benchmark-specific assumptions into the model.

---

# VERIFIER SYSTEM

Create a generic verifier interface.

Conceptually:

```
Verifier
   ├── exact match
   ├── mathematical
   ├── unit test
   ├── code execution
   └── custom
```

Only implement verifiers that can actually verify results.

---

# REASONING TRAINING

Create infrastructure for reasoning-oriented datasets.

Represent:

```
prompt
optional reasoning trace
answer
verifier result
```

Keep reasoning data separate from ordinary conversational data.

---

# REINFORCEMENT LEARNING ARCHITECTURE

Build a modular RL architecture.

Conceptually:

```
Task
  ↓
Policy model
  ↓
rollout
  ↓
environment
  ↓
verifier/reward
  ↓
trajectory
  ↓
trainer
  ↓
updated policy
```

Support asynchronous rollout architecture where practical.

Do not implement fake RL.

---

# AGENT SYSTEM

Build an agent abstraction capable of:

```
planning
tool selection
tool execution
observation processing
iterative reasoning
task completion
```

Represent trajectories explicitly.

Example:

```
task
message
action
tool_call
observation
action
reward
termination
```

---

# TOOL SYSTEM

Create a secure tool interface.

Potential tools:

```
filesystem
shell
Python
browser
database
custom APIs
```

Tools must have explicit permission boundaries.

Never grant unrestricted system access by default.

---

# SANDBOX

Agent execution must support sandboxing.

Provide:

```
filesystem restrictions
process restrictions
network policy
timeout
memory limits
CPU limits
```

Never assume tool execution is safe simply because the model requested it.

---

# LONG-HORIZON TASKS

Support multi-step environments.

Example:

```
task
  ↓
planning
  ↓
action
  ↓
observation
  ↓
correction
  ↓
action
  ↓
verification
  ↓
completion
```

Store complete trajectories.

---

# MULTIMODAL ARCHITECTURE

Design the model interfaces so additional modalities can be integrated.

Architecture should allow:

```
text
image
audio
video
```

without rewriting the entire Transformer.

Use modality adapters/projectors where appropriate.

Do not implement placeholder multimodality.

Only expose a modality when it has an actual working implementation.

---

# MODEL REGISTRY

Create a model registry.

Each architecture configuration should identify:

```
architecture type
version
dimensions
tokenizer
attention
MoE
context
precision
```

This makes checkpoints portable and inspectable.

---

# CONFIGURATION SYSTEM

The configuration system must support:

```
inheritance
profiles
overrides
validation
environment-variable overrides where appropriate
```

Do not duplicate huge YAML files.

Use a common base plus profile overrides.

---

# yami_super.yaml

Create:

```
configs/yami_super.yaml
```

It should define profiles approximately covering:

```
tiny
small
medium
500m
1b
3b
7b
14b
30b
70b
100b
300b
1t
```

These are architectural targets.

They must NOT be fake parameter labels.

The actual parameter calculator determines the resulting parameter count.

---

# TRILLION-PARAMETER SUPPORT

The framework must be architecturally capable of describing trillion-scale models.

A trillion-parameter configuration must be possible through combinations of:

```
large hidden dimensions
many layers
MoE
large expert counts
large expert dimensions
shared experts
```

The framework must NOT attempt to instantiate such a model during ordinary tests.

Instead support:

```
meta construction
parameter estimation
memory estimation
distributed topology validation
```

A trillion-parameter configuration must be able to be inspected without requiring trillion-parameter hardware.

---

# NO ARTIFICIAL LIMITS

Do not write code like:

```
if num_layers > 200:
    error
```

or:

```
if num_experts > 128:
    error
```

unless there is a genuine mathematical/runtime reason.

Hardware limitations should be detected dynamically.

---

# HARDWARE-DEPENDENT SCALING

The same configuration should be able to run differently depending on hardware.

Example:

```
development machine
    ↓
50M model

single GPU
    ↓
1B model

multi-GPU
    ↓
7B / 14B

multi-node cluster
    ↓
70B+

large cluster
    ↓
100B–1T+
```

Do not hard-code these associations.

They are examples of intended scalability.

---

# AUTO PARALLELISM

Design a planner:

```
yami distributed plan --config ...
```

Given:

```
model size
available GPUs
GPU memory
world size
```

calculate a possible parallelism topology.

Example output:

```
Data Parallel: 2
Tensor Parallel: 4
Pipeline Parallel: 2
Expert Parallel: 4
```

The planner must validate that:

```
DP × TP × PP
```

and any expert-parallel topology are compatible with world size.

Do not automatically launch training without explicit user confirmation.

---

# PERFORMANCE BENCHMARKS

Create benchmarks for:

```
forward pass
backward pass
attention
MoE
token throughput
memory consumption
inference latency
KV cache
distributed communication
```

Compare:

```
reference implementation
optimized implementation
```

where both exist.

---

# TESTING STRATEGY

Every subsystem must have unit tests.

Required tests:

```
config validation
parameter estimation
memory estimation
model construction
dense model
MoE model
GQA
RoPE
QK-Norm
SwiGLU
sliding-window attention
sparse attention
hybrid attention
KV cache
tokenizer compatibility
checkpoint save/load
checkpoint resume
inference
generation
distributed topology validation
expert routing
load balancing
```

Use tiny dimensions for tests.

Never instantiate enormous models during unit tests.

---

# PROFILE SMOKE TESTS

Every model profile must have a lightweight construction test.

Test:

```
tiny
small
medium
500m
1b
3b
7b
14b
30b
70b
100b
300b
1t
```

Use meta/device construction or parameter estimation where possible.

Do not allocate full large models.

---

# CLI

Provide clear commands such as:

```
yami model inspect
yami model validate
yami model estimate-params
yami model feasibility
yami hardware inspect
yami distributed plan
yami train
yami evaluate
yami generate
yami benchmark
yami checkpoint inspect
```

Follow the existing YaMi CLI style where possible.

---

# OBSERVABILITY

Provide detailed logging for:

```
model architecture
parameter count
active parameters
memory estimates
device placement
distributed topology
training progress
throughput
expert routing
```

Do not flood logs during normal inference.

Use configurable verbosity.

---

# ERROR HANDLING

Errors must be actionable.

Bad:

```
Configuration error.
```

Good:

```
num_attention_heads=48 is not divisible by num_kv_heads=7.
Choose a num_kv_heads value that divides 48.
```

Never silently change user configuration.

---

# DOCUMENTATION

Update the documentation comprehensively.

Explain:

```
architecture
parameter scaling
dense models
MoE
attention
long context
training
distributed training
hardware planning
checkpointing
inference
quantization
evaluation
post-training
RL
agents
multimodality
limitations
```

Include examples from:

```
50M
150M
1B
7B
30B MoE
70B MoE
1T MoE
```

Make clear which examples are:

```
tested
theoretically supported
hardware-dependent
```

---

# ENGINEERING QUALITY

Prefer:

```
clear interfaces
type safety
deterministic behavior
testability
performance
memory efficiency
modularity
```

Avoid:

```
unnecessary abstractions
duplicated code
hidden global state
magic constants
silent fallbacks
fake features
untested optimization
```

---

# IMPLEMENTATION ORDER

Follow this sequence.

## STEP 1 — Repository audit

Inspect the entire repository.

Produce an internal architecture map.

Identify:

```
existing model
attention
MoE
tokenizer
training
optimizer
scheduler
data
checkpoint
inference
evaluation
CLI
configuration
tests
```

Reuse what already works.

---

## STEP 2 — Configuration architecture

Implement:

```
yami_super.yaml
```

and scalable profile/override support.

---

## STEP 3 — Parameter engine

Implement accurate:

```
parameter estimation
active parameter calculation
memory estimation
```

before attempting huge models.

---

## STEP 4 — Architecture refactor

Implement modular:

```
attention
FFN
MoE
normalization
RoPE
KV cache
```

without breaking existing models.

---

## STEP 5 — Efficient kernels

Integrate available optimized implementations.

Keep correct reference implementations.

---

## STEP 6 — Distributed training

Implement genuine:

```
data parallelism
tensor parallelism
pipeline parallelism
expert parallelism
```

in an incremental, testable manner.

---

## STEP 7 — Data system

Implement:

```
streaming
sharding
mixtures
packing
deterministic sampling
```

---

## STEP 8 — Training system

Separate:

```
pretraining
SFT
preference training
RL
```

---

## STEP 9 — Inference

Implement scalable:

```
KV cache
batching
streaming
model parallel inference
```

---

## STEP 10 — Evaluation

Create comprehensive evaluation infrastructure.

---

## STEP 11 — Agent/RL

Implement:

```
environments
trajectories
verifiers
tools
reward interfaces
```

---

## STEP 12 — Multimodal interfaces

Only implement actual working modalities.

---

## STEP 13 — Billion-scale validation

Validate:

```
1B
3B
7B
14B
```

architectures using appropriate hardware.

---

## STEP 14 — Large-scale validation

Validate configuration and distributed planning for:

```
30B
70B
100B
300B
1T
```

without requiring full local instantiation.

---

# FINAL AUDIT

Before declaring completion, inspect the entire codebase again.

Search for:

```
TODO
FIXME
pass
NotImplemented
placeholder
mock
fake
hard-coded limits
ignored configuration
unreachable configuration
duplicate implementations
```

Fix anything relevant.

Verify that every configuration field is actually connected to executable code.

---

# FINAL REPORT

After implementation, report:

1. Files changed.
2. Files created.
3. Existing functionality preserved.
4. New architecture features.
5. New attention features.
6. MoE capabilities.
7. Distributed capabilities.
8. Parameter estimator results.
9. Memory estimator results.
10. Supported model profiles.
11. Maximum model size actually tested.
12. Maximum model size theoretically configurable.
13. Maximum context actually tested.
14. Maximum context theoretically configurable.
15. Hardware requirements.
16. Training capabilities.
17. Inference capabilities.
18. Evaluation capabilities.
19. RL capabilities.
20. Agent capabilities.
21. Multimodal capabilities.
22. Quantization capabilities.
23. Tests executed.
24. Benchmark results.
25. Known limitations.
26. Features still requiring future implementation.

Never claim that a large parameter count alone makes the model intelligent.

A trillion-parameter configuration is an engineering capability.

Actual model capability depends on:

```
architecture
training data
training compute
optimization
training duration
post-training
reinforcement learning
evaluation
inference
tool use
model quality
```

---

# DEFINITION OF DONE

The task is NOT complete when:

```
yami_super.yaml exists.
```

The task is complete only when:

```
configuration
    ↓
architecture
    ↓
parameter calculation
    ↓
memory estimation
    ↓
model construction
    ↓
training
    ↓
distributed execution
    ↓
checkpointing
    ↓
inference
    ↓
evaluation
```

form one coherent, tested system.

The ultimate design principle is:

```
SMALL HARDWARE
     ↓
  small YaMi

MORE HARDWARE
     ↓
  larger YaMi

MULTI-GPU
     ↓
  billion-scale YaMi

MULTI-NODE CLUSTER
     ↓
  10B–100B+ YaMi

LARGE TRAINING CLUSTER
     ↓
  100B–1T+ YaMi
```

The software must not artificially prevent scaling.

Hardware, memory, communication bandwidth, training data, and compute should determine practical limits—not arbitrary constants in the code.

Build YaMi as a **scalable LLM platform**, not a single fixed-size model.
