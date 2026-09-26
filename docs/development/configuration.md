# Configuration System

## Hierarchy

```
defaults (schema dataclass defaults, tiny model)
  └── config YAML files (--config, repeatable; later files win per key)
        └── section aliases (--model-config / --training-config / --data-config / --inference-config)
              └── --set section.key=value overrides (highest priority)
```

```
configs/
├── model/       tiny.yaml · small.yaml · medium.yaml · coding_low.yaml · coding_mid.yaml · coding_high.yaml
├── training/    local.yaml · tiny_test.yaml · kaggle.yaml
├── data/        default.yaml
└── inference/   default.yaml
```

## Rules

- **Typed schemas** — each section binds to a dataclass
  (`fontaine/config/schema.py`); unknown keys fail with the dotted path
  (typo protection), values are type-coerced, and cross-field validation runs
  once at load (e.g. `hidden_size % num_attention_heads == 0`,
  `data.sequence_length ≤ model.max_sequence_length`).
- **`model.vocab_size: auto`** — resolved from the trained tokenizer at train
  time; explicit ints are checked against the tokenizer instead.
- **No hidden config** — every knob is a schema field with a default; the
  *resolved* config is snapshotted into `experiments/<run>/config.yaml` and
  into every checkpoint's `meta.json`.

## Examples

```bash
# canonical form (as in the spec)
python -m fontaine train \
  --config configs/model/tiny.yaml \
  --training-config configs/training/local.yaml \
  --data-config configs/data/default.yaml \
  --tokenizer-dir datasets/tokenizer

# override anything, at any depth, without editing files
python -m fontaine train --model-config configs/model/tiny.yaml \
  --training-config configs/training/local.yaml \
  --tokenizer-dir datasets/tokenizer \
  --set training.max_steps=100000 \
  --set model.num_layers=8 \
  --set 'data.raw_paths=["datasets/raw/a.txt", "datasets/raw/b.jsonl"]'

# JSON values work in overrides
--set training.early_stopping_patience=5 --set inference.seed=null
```

## Adding a new subsystem

Add a dataclass + a section key in `config/loader.py` (`_SECTION_TYPES`) and
it participates in merging, validation, snapshots, and `--set` immediately.
