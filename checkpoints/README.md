# checkpoints/

Standalone checkpoints live here but are **never committed to Git**.

Training runs write checkpoints into their own experiment directory
(`experiments/<run>/checkpoints/`); use this folder only for checkpoints you
manually promote out of experiments (e.g. a release model you want to keep
when an experiment is pruned).

Format: `docs/training/checkpoint-format.md`.
