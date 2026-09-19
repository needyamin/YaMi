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
| `architecture` | whole model | extension point; `decoder_transformer` today |
| `vocab_size` | embedding + head | int or `"auto"` (resolve from tokenizer) |
| `hidden_size`, `num_layers` | capacity | scale together (Chinchilla-style) |
| `num_attention_heads` / `num_kv_heads` | attention | ratio = GQA group size |
| `intermediate_size` | FFN | 3×hidden (SwiGLU) or 4×hidden (GELU) |
| `max_sequence_length` | RoPE table + KV cache | context window |
| `rope_theta` | RoPE | context-length scaling lever |
| `dropout` | attention + residuals | 0.0 for LLM pretraining typical |
| `activation`, `normalization` | FFN, blocks | swiglu/gelu, rmsnorm/layernorm |
| `tie_word_embeddings` | head | parameter savings at small scale |

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
- **Mixture-of-experts**: multiplies infra complexity; the FFN is a clean
  swap-in point when compute allows.
