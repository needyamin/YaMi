# Reproducibility & Experiment Workflow

## What every experiment records

`experiments/<date>_<time>_<run_name>/`:

```
├── config.yaml        # fully resolved configuration
├── environment.json   # git commit · python/torch versions · platform · device · seed · start time
├── metrics.jsonl      # append-only, step-ordered metric records
├── summary.json       # duration + final metrics, written at run end
├── logs/              # full log output
└── checkpoints/       # CheckpointManager root (latest/best pointers, metadata)
```

Combined with the dataset manifest (tokenizer version, preprocessing version,
checksums) and checkpoint metadata, every number in `metrics.jsonl` is
traceable to a code commit + config + data version + seed.

## Determinism controls

- `training.seed` seeds python/numpy/torch (CUDA included when present).
- `training.deterministic: true` requests deterministic kernels (warn-only, so
  exotic ops don't crash the run).
- Data order: the DataLoader generator is seeded; shuffling is reproducible;
  resume restores RNG states captured at checkpoint time.
- Generation: `inference.seed` creates a per-request generator.

Full determinism on GPU is best-effort by nature (atomic ops, reduced-precision
accumulation); the goal is *bit-stable resume* and *statistically identical
reruns*, both of which hold.

## Workflow

```bash
# 0. plan resources before touching the GPU
fontaine model inspect --model-config configs/model/tiny.yaml

# 1. prepare data (validated, checksummed, versioned)
fontaine data prepare …

# 2. train → creates the experiment dir automatically
fontaine train …

# 3. inspect progress
cat experiments/*run/metrics.jsonl | tail

# 4. resume (same experiment dir, continuous metrics history)
fontaine train --resume experiments/<run>/checkpoints …

# 5. compare / promote
fontaine evaluate --checkpoint experiments/<run>/checkpoints/best …
```

## Debugging guide

| Symptom | First checks |
| --- | --- |
| loss NaN | lower LR; check for empty/garbage docs (`data validate`); `max_grad_norm` active; bf16/fp16 switch |
| loss flat | tokenizer/data mismatch (`vocab_size` check), seq length ≫ data structure, LR too small |
| OOM | `fontaine model inspect`; ↓ batch, ↑ grad accum, ↑ gradient checkpointing, ↓ seq len |
| val ≫ train loss | split leakage in reverse (val too small/hard) or distribution shift — check `tokens_per_split` |
| slow CPU training | that's expected for anything above Tiny; use GPU or shrink the model |
| checkpoint load fails | integrity error = corrupted file (re-save); format_version error = run `checkpoint convert` |

## git discipline

- Commit before runs (the commit hash lands in `environment.json`).
- Keep datasets/checkpoints out of Git; keep configs in Git.
- Datasets are named/versioned in their manifests — treat data versions like
  code versions.
