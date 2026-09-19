# Tokenizer Design

The tokenizer is **independent from the model**: the model only sees integer
ids and `vocab_size`. Any implementation of the `Tokenizer` interface can
drive training and inference.

## The interface (`src/fontaine/tokenizer/base.py`)

```python
class Tokenizer(ABC):
    vocab_size: int            # includes special tokens
    version: str               # stable hash — recorded in datasets/checkpoints
    encode(text, add_special_tokens=False) -> list[int]
    decode(ids, skip_special_tokens=True) -> str
    encode_batch(texts) -> list[list[int]]
    save(directory) / load(directory)
    pad_id / bos_id / eos_id / unk_id   # ids 0..3 by contract
```

Contract guarantees: ids `[0,1,2,3]` are always pad/bos/eos/unk; `version`
changes whenever the vocabulary changes and is recorded in every dataset
manifest, checkpoint, and experiment.

## Choices considered

| Option | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| Word-level | trivial | huge vocab, no OOV handling | reject |
| Character-level | tiny vocab, no unknowns, dependency-free | long sequences (worse compute per concept) | keep as **dev/test tokenizer** |
| BPE (byte-level) | compact vocab, no OOV (all bytes representable), best compute/quality trade-off | needs training infra | **recommended default** |
| SentencePiece Unigram | probabilistic, strong multilingual | extra dependency, heavier | future option behind the same interface |

## Recommendation

- **Development, tests, debugging:** `CharTokenizer` (`tokenizer.type: char`)
  — zero dependencies, trains instantly, makes the whole pipeline exercisable
  on any machine.
- **Real training:** `ByteLevelBPETokenizer` (`tokenizer.type: hf_bpe`) —
  trains from a *streaming* corpus iterator (never loads the corpus), uses the
  HF `tokenizers` Rust implementation (optional dependency isolated to
  `hf_bpe.py`), byte-level so no text is ever unencodable.

```bash
fontaine tokenizer train --tokenizer-dir datasets/tokenizer \
  --set tokenizer.type=hf_bpe --set tokenizer.vocab_size=8192 \
  --set 'data.raw_paths=["datasets/raw"]'
fontaine tokenizer test --tokenizer-dir datasets/tokenizer
```

## Replaceability

`fontaine/tokenizer/registry.py` is the only place that knows concrete
classes (`train_tokenizer`, `load_tokenizer`). Adding a SentencePiece
implementation = one new class + one registry branch. The model, data
pipeline, trainer, and inference engine require **zero changes**; only
`model.vocab_size` follows the tokenizer (`auto` handles this).

## Versioning discipline

- Tokenizer `version` is a content hash of the serialized vocabulary.
- Dataset manifests record `tokenizer.name/version/vocab_size`.
- Checkpoints record the tokenizer metadata; `fontaine train` refuses to
  pair a model with an incompatible tokenizer (`resolve_vocab_size` /
  `check_vocab_compatibility`), preventing silent vocab drift.
