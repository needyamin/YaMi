# Multimodal Architecture: Extension Points

Multimodality is **not implemented** — deliberately. The initial codebase
stays text-only; the seams below keep it open.

## Target shape

```
Image Encoder        Audio Encoder       Video Encoder
(ViT / CLIP-style)   (wav2vec / Whisper) (frame encoder + temporal pool)
        │                    │                    │
        └────────────┬───────┴────────────────────┘
                     ▼
      Multimodal Projection / Adapter
      (linear / MLP / Perceiver-resampler → hidden_size vectors)
                     │
                     ▼
     Language Model  (FontaineModel — unchanged decoder)
                     │
                     ▼
        Text / Multimodal output
```

## How it attaches with minimal disruption

1. **Encoders are separate models.** Each lives in its own package/module
   with its own checkpoint type; they are *not* part of `FontaineModel`.
2. **The projection layer is the adapter.** Encoder outputs are projected to
   `hidden_size` and inserted into the token sequence as *embedding
   replacements* at designated placeholder positions — the standard
   LLaVA-style approach. `FontaineModel.forward` gains an optional
   `inputs_embeds`-style path (embedding lookup bypass); everything downstream
   (attention stack, KV cache, generation loop) is already
   sequence-agnostic and needs no change.
3. **Position handling.** Encoded chunks occupy consecutive positions; RoPE
   applies uniformly. Audio/video temporal pooling determines the number of
   positions per sample — a dataset/packing concern, not a model one.
4. **Data contract.** The manifest/shard format grows a parallel media-shard
   index (paths + offsets); text tokens gain placeholder token ids (new
   special tokens — the tokenizer contract supports registry-defined
   specials).
5. **Training recipe.** Stage 1: train the projection adapter with the LM
   frozen (cheap, stable). Stage 2: unfreeze and continue pretraining with
   interleaved data.

## What NOT to do now

- Don't add encoder stubs or config fields that pretend to exist.
- Don't entangle the current text pipeline with future media handling — the
  clean seam is exactly why the extension will be cheap later.

The single forward-signature change (`inputs_embeds` passthrough) is the only
planned modification to the model — everything else lands in new modules.
