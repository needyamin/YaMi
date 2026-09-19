"""Full end-to-end test through the CLI: tokenizer -> data -> train -> generate -> evaluate."""

import glob

import pytest

from fontaine.cli.main import main

pytestmark = pytest.mark.integration


def _run(argv):
    code = main(argv)
    assert code == 0, f"command failed: {argv}"


def test_full_pipeline_through_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    corpus = tmp_path / "raw"
    corpus.mkdir()
    (corpus / "data.txt").write_text(
        "".join(
            f"the fontaine sings over stone {i}\nand the water answers\n\n"
            for i in range(50)
        ),
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"

    _run([
        "tokenizer", "train",
        "--tokenizer-dir", str(tokenizer_dir),
        "--set", f'data.raw_paths=["{corpus.as_posix()}"]',
        "--set", "tokenizer.type=char",
    ])

    _run([
        "data", "prepare",
        "--tokenizer-dir", str(tokenizer_dir),
        "--set", f'data.raw_paths=["{corpus.as_posix()}"]',
        "--set", f'data.output_dir="{(tmp_path / "prepared").as_posix()}"',
        "--set", "data.sequence_length=32",
        "--set", "data.min_document_chars=4",
        "--name", "cli-dataset",
    ])

    _run([
        "train",
        "--tokenizer-dir", str(tokenizer_dir),
        "--set", "model.vocab_size=auto",
        "--set", "model.hidden_size=64",
        "--set", "model.num_layers=2",
        "--set", "model.intermediate_size=192",
        "--set", "model.max_sequence_length=64",
        "--set", 'data.manifest_path="prepared/manifest.json"',
        "--set", "data.sequence_length=32",
        "--set", "training.max_steps=30",
        "--set", "training.batch_size=8",
        "--set", "training.eval_interval=30",
        "--set", "training.checkpoint_interval=30",
        "--set", "training.warmup_steps=2",
        "--set", "training.run_name=cli-test",
        "--set", 'training.experiments_dir="experiments"',
    ])

    checkpoints = sorted(
        glob.glob(str(tmp_path / "experiments" / "*cli-test" / "checkpoints" / "step_*"))
    )
    assert checkpoints
    checkpoint = checkpoints[-1]

    _run([
        "generate",
        "--checkpoint", checkpoint,
        "--tokenizer-dir", str(tokenizer_dir),
        "--prompt", "the fontaine",
        "--max-new-tokens", "10",
        "--set", "inference.temperature=0.0",
    ])

    _run([
        "evaluate",
        "--checkpoint", checkpoint,
        "--tokenizer-dir", str(tokenizer_dir),
        "--set", 'data.manifest_path="prepared/manifest.json"',
        "--set", "training.batch_size=8",
        "--set",
        'evaluation.evaluators=[{"name": "validation_loss", "params": {"max_batches": 4}}]',
    ])

    _run(["checkpoint", "inspect", "--checkpoint", checkpoint])
