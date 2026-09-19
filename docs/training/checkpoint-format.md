# Checkpoint Format Specification

## Layout (format_version 1)

```
<root>/                              # e.g. experiments/<run>/checkpoints/
├── step_000000200/
│   ├── model.pt                     # {"model": state_dict}  (fp32 master weights)
│   ├── optimizer.pt                 # {"optimizer": …, "scheduler": …, "scaler": …}
│   ├── rng.pt                       # python / numpy / torch RNG states
│   └── meta.json                    # everything below
├── latest.json                      # {"step": 200, "path": "step_000000200"}
└── best.json                        # {"step": 150, "path": "step_000000150",
                                     #  "metric": {"name": "val_loss", "value": 0.91}}
```

`meta.json`:

| Field | Meaning |
| --- | --- |
| `format_version` | 1 — loaders refuse anything else |
| `step`, `epoch` | training progress |
| `created_utc` | ISO timestamp |
| `world_size` | 1 today; sharded checkpoints record fan-out |
| `config` | full resolved config snapshot (model truth lives here) |
| `tokenizer` | `{name, version, vocab_size}` |
| `dataset` | manifest path, name/version, tokenizer/preprocessing versions |
| `code` | Fontaine version (+ git commit via experiment env) |
| `metric` | tracked metric for best-checkpoint logic |
| `files` | `{filename: sha256}` of every file in the checkpoint |

## Guarantees

- **Atomicity** — every file is written to a temp name and moved into place;
  a crash mid-save never produces a half checkpoint.
- **Integrity** — `load()` recomputes SHA-256 of each file and fails loudly
  on mismatch (`CheckpointIntegrityError`), so corruption is caught at resume
  time, not mid-run.
- **No symlinks** — `latest`/`best` are JSON pointer files (Windows-safe).
- **Complete resume** — weights + optimizer + scheduler + scaler + RNG states
  + step counter reproduce the training stream exactly.
- **Pruning with protection** — `keep_last_n` newest step dirs are kept plus
  the best-checkpoint directory.

## Checkpoint contents checklist

model weights ✓ optimizer state ✓ scheduler state ✓ scaler state (if any) ✓
training step ✓ epoch ✓ random states ✓ configuration ✓ tokenizer version ✓
dataset version ✓ code/version metadata ✓ integrity hashes ✓

## Inference loading

`load_generator(checkpoint, tokenizer_dir)` rebuilds the architecture from the
stored `config.model` snapshot — a checkpoint is self-describing; you never
need to remember which YAML produced it.

## Conversion

`fontaine checkpoint convert --checkpoint <dir> --format safetensors` exports
weights (requires the optional `safetensors` package; integrity is verified
before export). Future targets (HF format, sharded layouts) register in
`checkpointing/convert.py`.

## Future: sharded checkpoints

The v1 layout anticipates distribution:
- `world_size` is recorded from day one.
- Evolution: `step_XXXXXX/rank_00/model.pt … rank_NN/` with a global
  `meta.json` index — same pointers, same integrity model, same manager API.
- FSDP/DeepSpeed state dicts flatten into the same `model.pt` structure
  (full-state-dict APIs) before persisting, keeping the format
  parallelism-agnostic.
