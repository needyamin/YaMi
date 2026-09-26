# Model Design

`FontaineModel` (`src/fontaine/models/transformer.py`) is a modern
decoder-only Transformer. One class serves every size; sizes are values of
`ModelConfig` (see `configs/model/`), never code.

```mermaid
flowchart TB
    T[Token ids] --> E[Token Embedding]
    E --> B1["Transformer Block 1<br/>RoPE · GQA · SwiGLU · RMSNorm"]
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
- Training uses fused causal `scaled_dot_product_attention`; incremental
  decoding uses an explicit mask against the cached prefix.

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
Quantization, paged KV, and speculative decoding slot in behind this
interface later.

## Why not something else?

- **Encoder-decoder**: unnecessary for autoregressive LM pretraining; SFT on
  the same decoder stack is the standard path.
- **State-space models (Mamba)**: promising, but the Transformer is the
  well-understood default; the `architecture` field leaves room to add one.
- **Mixture-of-experts as a separate model class**: the FFN is the swap.
  `moe_decoder` is the same `FontaineModel` with a routed feed-forward, so
  the trainer, checkpoints, and KV cache stay unchanged.
