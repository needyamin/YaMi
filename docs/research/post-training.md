# Post-Training: Extension Points

The pretraining stack is the foundation; the instruction/alignment layers are
designed for, not yet implemented. The path:

```
Pretraining → Supervised Fine-Tuning → Preference Optimization → Evaluation → Deployment
```

## Why the current design anticipates this

- **Loss masking exists now.** The model's loss ignores `labels == -100`, and
  the dataset pipeline writes label windows. SFT = instruction-format JSONL
  (already partially parsed by `sources.py`: `prompt`/`response` records) +
  a collator that masks prompt tokens. No model or trainer change.
- **`Trainer` is component-agnostic.** SFT reuses `Trainer` verbatim with a
  different dataset; DPO/RL add *new trainer classes* composing the same
  collaborators (model, optimizer, checkpointing, evaluation, logging).
- **Evaluation registry** hosts instruction-following benchmarks the moment
  an SFT checkpoint exists.

## Planned components

| Component | What it does | New code needed |
| --- | --- | --- |
| Instruction tuning | next-token training on prompt-masked instruction data | data adapter + collator; `Trainer` as-is |
| Supervised fine-tuning (SFT) | domain/style adaptation on curated pairs | same as above; lower LR configs |
| Preference datasets | paired responses with human ratings | new manifest-backed dataset type |
| DPO-style optimization | offline preference optimization from pairs | `DPOTrainer` (two models / ref frozen) + loss |
| Reward modeling | scalar reward on comparisons | small head on `FontaineModel` + trainer |
| RL-based post-training | on-policy improvement vs reward | rollout loop using `Generator` (KV cache already supports it), PPO/GRPO-style trainer |

## Sequencing guidance

1. Ship SFT first (highest value per engineering hour; pure data + masking).
2. Then preference optimization (DPO is offline and stable relative to RL).
3. Reward modeling only when a preference dataset exists to train it on.
4. RL last — it consumes everything above plus significant compute.

Each stage produces ordinary checkpoints (same format/registry flow), so
deployment tooling never learns about post-training specifics.
