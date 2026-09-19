# Model Registry (design)

The registry answers one question reliably: **"which exact model is deployed
or being evaluated, and where did it come from?"** Today the information
already exists — in checkpoints (`meta.json`) and experiment directories —
but it is not indexed. The registry is the index.

## Record schema

| Field | Source today |
| --- | --- |
| model name + version (`Fontaine-Tiny-v0.1`) | run name + checkpoint step |
| architecture + full config | checkpoint `meta.config` |
| parameter count | analytic from config |
| tokenizer (name/version/vocab) | checkpoint `meta.tokenizer` |
| dataset (name/version/license) | checkpoint `meta.dataset` + manifest |
| training configuration | experiment `config.yaml` |
| evaluation results | experiment `metrics.jsonl` / `summary.json` |
| checkpoint location(s) | checkpoint pointers (`latest`/`best`) |
| status | `training → candidate → evaluated → deployed / retired` |
| compatibility | Fontaine version, checkpoint format version |

## Storage evolution

- **Phase 1 (now, zero infra):** convention-based — `experiments/<run>/` +
  `meta.json` is the record; `fontaine checkpoint inspect` reads it.
- **Phase 2:** `registry.jsonl` / SQLite index built by scanning experiment
  dirs; a `fontaine registry list|show` command.
- **Phase 3:** service (or an existing one — MLflow, W&B artifacts) with
  staged promotions and deployment binding.

## Naming convention

```
Fontaine-<Size>-v<MAJOR.MINOR>
Fontaine-Tiny-v0.1   # first trained tiny model
Fontaine-Tiny-v0.2   # same architecture, more data/steps
Fontaine-Small-v1.0  # first "release-quality" small model
```

Promotion rules (suggested, project-specific): a version may be marked
`deployed` only when its recorded evaluation results pass the current
benchmark thresholds and its checkpoint integrity verifies.
