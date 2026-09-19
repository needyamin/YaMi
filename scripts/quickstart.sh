#!/usr/bin/env bash
# Fontaine AI quickstart: tokenizer -> data -> tiny training -> generation.
# Uses a tiny synthetic corpus so it runs in ~1-2 minutes on any 16 GB machine.
set -euo pipefail

RAW_DIR="datasets/quickstart/raw"
TOK_DIR="datasets/quickstart/tokenizer"
DATA_DIR="datasets/quickstart/prepared"

# 1. tiny synthetic corpus (replace datasets/quickstart/raw with your data)
mkdir -p "$RAW_DIR"
python - <<'PY'
import pathlib, random
random.seed(7)
words = ["fontaine", "water", "spring", "river", "ocean", "model", "train",
         "data", "token", "small", "scale", "dream", "light", "stone", "music", "code"]
out = pathlib.Path("datasets/quickstart/raw")
for i in range(60):
    doc = " ".join(random.choice(words) for _ in range(40))
    (out / f"stories_{i}.txt").write_text(doc + "\n\n" + doc.lower() + "\n\n", encoding="utf-8")
print("synthetic corpus ready in", out)
PY

# 2. tokenizer
python -m fontaine tokenizer train --tokenizer-dir "$TOK_DIR" \
  --set tokenizer.type=hf_bpe --set tokenizer.vocab_size=1024 \
  --set "data.raw_paths=[\"$RAW_DIR\"]" --set data.min_document_chars=16

# 3. data preparation
python -m fontaine data prepare --tokenizer-dir "$TOK_DIR" \
  --set "data.raw_paths=[\"$RAW_DIR\"]" \
  --set "data.output_dir=$DATA_DIR" --set data.sequence_length=128 \
  --set data.min_document_chars=16 --name quickstart --license "CC0-1.0"

# 4. short tiny-model training
python -m fontaine train --model-config configs/model/tiny.yaml \
  --training-config configs/training/tiny_test.yaml \
  --tokenizer-dir "$TOK_DIR" \
  --set "data.manifest_path=\"$DATA_DIR/manifest.json\"" \
  --set data.sequence_length=128 --set model.max_sequence_length=256 \
  --set training.run_name=quickstart

RUN_DIR=$(ls -dt experiments/*quickstart | head -1)
CKPT=$(ls -d "$RUN_DIR"/checkpoints/step_* | sort | tail -1)

# 5. generate
python -m fontaine generate --checkpoint "$CKPT" --tokenizer-dir "$TOK_DIR" \
  --prompt "the fontaine" --max-new-tokens 40 --set inference.temperature=0.8

echo "quickstart complete — experiment: $RUN_DIR"
