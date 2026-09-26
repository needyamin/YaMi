# Testing Strategy

116 tests across seven areas; `python -m pytest` runs the whole suite in a few
minutes on CPU with zero network access and no committed test data.

```
tests/
├── unit/           config · sampling · memory · utils
├── model/          components · forward/backward · mixture of experts
├── tokenizer/      char + BPE
├── data/           pipeline · shards · manifest
├── training/       trainer · checkpointing
├── inference/      generation engine
└── integration/    full CLI pipeline
```

## What is verified

| Area | Checks |
| --- | --- |
| tokenizer correctness | round-trips, special-token id contract, unknown handling, save/load identity, version stability, collision guards |
| tensor shapes | every component and the full model, GQA and MHA, tied and untied |
| attention correctness | RoPE relative-distance property, causality (future tokens cannot affect past logits), KV-cache ≡ full forward |
| model forward/loss | shapes, finiteness, `-100` ignore-index semantics |
| gradient flow | dense models: every trainable parameter receives a gradient; mixture-of-experts: the router learns, and an idle expert on a tiny batch may see no tokens; checkpointing preserves loss |
| parameter counting | analytic estimator ≡ instantiated model (two independent implementations must agree) |
| checkpoint save/load | exact weight restore, metadata/provenance, best/latest pointers, pruning keeps best, integrity detects corruption, RNG resume exactness |
| dataset loading | manifest validation, checksum tamper detection, window shifting, lazy memmap, determinism of preparation |
| generation | greedy/seeded determinism, max-new-tokens, stop sequences, stream-deltas ≡ full text, context budget (keep prompt and answer when both fit; otherwise cap the prompt at 75% of the window) |
| configuration validation | unknown keys/sections, type coercion, cross-field rules |
| **overfit sanity check** | the tiny model genuinely overfits a tiny synthetic dataset (loss falls >60%, perplexity < 4) |
| end-to-end | `tokenizer train → data prepare → train → generate → evaluate → checkpoint inspect` through the real CLI in a temp dir |

## The tiny synthetic dataset

`tests/conftest.py` generates a repetitive, highly predictable corpus (fixed
template sentences with counters) — no data files in Git, no randomness in
structure. It is small enough that the whole trainer can overfit it, which is
the cheapest end-to-end proof that *model + data + loss + optimizer +
checkpointing* all work together.
