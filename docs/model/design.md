# Model Design

`FontaineModel` (`src/fontaine/models/transformer.py`) is a modern
decoder-only Transformer. One class serves every size; sizes are values of
`ModelConfig` (see `configs/model/`), never code.

```mermaid
flowchart TB
    T[Token ids] --> E[Token Embedding]
    E --> B1["Transformer Block 1<br/>RoPE · GQA · QK-norm · SwiGLU · RMSNorm"]
    B1 --> B2["Transformer Block 2"]
    B2 --> BN["… × num_layers"]
    BN --> FN[Final RMSNorm]
    FN --> H[LM Head<br/>(tied with embedding)]
    H --> L[Logits / Loss]
```

## Components and why each exists

### Token embedding
Maps token ids to `hidden_size` vectors. `tie_word_embeddings: true` shares
one matrix between input embedding and output head — saves
`vocab_size × hidden_size` parameters and improves quality at small scale.
Untie when `hidden_size` grows and the two roles diverge.

### Positional encoding: RoPE
Rotary embeddings (`components.build_rope_cache`) rotate Q/K by an angle
proportional to position, so attention scores depend on *relative* distance.
Chosen because: it extrapolates better than learned absolute positions, it is
the standard for modern LLMs, and `rope_theta` is the primary lever for
context-length extension (raise theta / fine-tune on longer windows — NTK/YaRN
style — without architectural change).

### Self-attention with GQA
`CausalSelfAttention` supports `num_kv_heads ≤ num_attention_heads`
(grouped-query attention): query heads are grouped to share K/V heads.
- Cuts KV-cache memory/bandwidth by the head ratio — decisive for long
  context and inference throughput.
- `num_kv_heads == num_attention_heads` degrades to standard MHA.
- The cache stores only the compact shared K/V (verified by tests).
- `scaled_dot_product_attention` reads the shared K/V heads directly
  (`enable_gqa`) when torch supports it, so K/V are never copied per query
  head. Older torch falls back to `repeat_interleave`.
- A block starting at position 0 uses fused causal attention. A single
  decode token on a full-attention layer needs no mask. Sliding-window
  layers and chunked prefill build a boolean mask.

### QK-norm
`qk_norm: true` applies an RMSNorm to each query and key head before RoPE,
as in Qwen3, Gemma 3, and OLMo 2. It bounds attention logits, which keeps
training stable at higher learning rates and in bf16. It costs
`2 × head_dim` weights per layer.

### Sliding-window attention
`sliding_window: N` limits a layer to the last N tokens. With
`global_attention_every: k`, every k-th layer keeps full attention, as in
Gemma 3 and Mistral. Local layers cost O(N) per token instead of
O(context), while the global layers still carry long-range information. The
KV cache is still sized for the full window.

### RoPE scaling
`rope_scaling_type: linear` divides positions by `rope_scaling_factor`
(position interpolation). `ntk` raises the wavelength base by
`factor ** (d / (d - 2))`, which leaves the fastest dimensions untouched.
Both let a checkpoint run past its trained context after a short fine-tune.
Raise `max_sequence_length` to the new window at the same time.

### Feed-forward: SwiGLU
`down(silu(gate(x)) * up(x))` — the gated FFN used by Llama-class models.
Configurable `intermediate_size` (≈2.7×hidden for SwiGLU vs ≈4×hidden for
GELU); `activation: gelu` selects the classic MLP if wanted.

### Mixture of experts
`architecture: moe_decoder` with `num_experts >= 2` replaces that MLP with a
token-choice mixture (`MixtureOfExperts`). A bias-free router picks the top
`num_experts_per_token` experts, and only those experts run, only on the
tokens that selected them. **Active** parameters are what one token
multiplies (attention, norms, router, and the chosen experts, plus the
embedding). **Total** parameters count every expert — AdamW stores all of
them, so training memory follows the total. A dense model
(`num_experts: 1`) has active equal to total and no router.

Training adds a Switch load-balance term,
`num_experts * sum(fraction_dispatched * mean_router_prob)`, scaled by
`moe_aux_loss_coef` (default 0.01). The dispatch fraction is detached.
The forward pass sorts (token, expert) pairs by expert once, then runs each
expert on one contiguous slice, so a step needs one host sync in total.
Evaluation and generation do not add the term, so validation loss stays
plain cross-entropy. The coding ladder
(`configs/model/coding_low.yaml`, `coding_mid.yaml`, `coding_high.yaml`)
uses this so a laptop can run a model whose stored capacity is larger than
the matmuls per token.

### Normalization: RMSNorm (pre-norm)
Pre-norm residual blocks (`x + attn(norm(x))`) keep gradients healthy in deep
stacks. RMSNorm is cheaper than LayerNorm and equally stable; computed in
fp32 for numerical safety under autocast. `normalization: layernorm` is
configurable.

### Residual connections
Two per block (attention, FFN). Residual output projections are initialized
at `0.02 / √(2·num_layers)` so deep stacks start near identity — standard
GPT-2-style initialization discipline.

### Output head + loss
`lm_head` projects to `vocab_size`. Loss is cross-entropy with
`ignore_index=-100`, chosen so future SFT can mask prompt tokens without any
model change. `ModelOutput` carries both logits and (optionally) loss.

## Configurability map

| Field | Component | Notes |
| --- | --- | --- |
| `architecture` | whole model | `decoder_transformer` or `moe_decoder` |
| `vocab_size` | embedding + head | int or `"auto"` (resolve from tokenizer) |
| `hidden_size`, `num_layers` | capacity | scale together (Chinchilla-style) |
| `num_attention_heads` / `num_kv_heads` | attention | ratio = GQA group size |
| `intermediate_size` | FFN | 3×hidden (SwiGLU) or 4×hidden (GELU) |
| `max_sequence_length` | RoPE table + KV cache | context window |
| `rope_theta` | RoPE | context-length scaling lever |
| `rope_scaling_type`, `rope_scaling_factor` | RoPE | `none`, `linear`, or `ntk` context extension |
| `qk_norm` | attention | RMSNorm on query and key heads |
| `sliding_window`, `global_attention_every` | attention | local window; every k-th layer global |
| `dropout` | attention + residuals | 0.0 for LLM pretraining typical |
| `activation`, `normalization` | FFN, blocks | swiglu/gelu, rmsnorm/layernorm |
| `tie_word_embeddings` | head | parameter savings at small scale |
| `num_experts` | FFN | 1 = dense MLP; ≥2 = mixture of experts |
| `num_experts_per_token` | router | how many experts run per token (active params) |
| `moe_aux_loss_coef` | training loss | Switch load-balance term; 0 disables it |
| `cpu_tier` | inspect only | label printed by `fontaine model inspect` |

## Inference path

The model accepts an optional `KVCache` (`models/kv_cache.py`): pre-allocated
K/V tensors sized `max_sequence_length`. Cache-incremental logits are
verified against the full forward pass (`test_kv_cache_matches_full_forward`).

`inference.precision` (or `--precision` on `generate` and `serve`) picks the
weight format at load time (`optimization/quantize.py`):

| Value | What happens |
| --- | --- |
| `auto` | int8 on CPU for 20M+ parameters, fp32 for smaller models, bf16 on a GPU that supports it |
| `int8` | dynamic int8 quantization of every Linear (CPU only); embeddings and norms stay fp32 |
| `bf16` | bf16 weights and KV cache; falls back to fp32 on a CPU without AVX-512 BF16 or AMX |
| `fp32` | full precision (always used by `fontaine evaluate`) |

`inference.num_threads` (or `--threads`) sets the CPU thread count; 0 keeps
the PyTorch default. The server reports the format as `quantization_level`
(`F32`, `BF16`, `Q8_0`). Paged KV and speculative decoding slot in behind
this interface later.

## Why not something else?

- **Encoder-decoder**: unnecessary for autoregressive LM pretraining; SFT on
  the same decoder stack is the standard path.
- **State-space models (Mamba)**: promising, but the Transformer is the
  well-understood default; the `architecture` field leaves room to add one.
- **Mixture-of-experts as a separate model class**: the FFN is the swap.
  `moe_decoder` is the same `FontaineModel` with a routed feed-forward, so
  the trainer, checkpoints, and KV cache stay unchanged.
