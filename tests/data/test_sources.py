"""Raw-data readers: every format, archives, compression, and an end-to-end train check."""

import bz2
import csv
import gzip
import io
import json
import lzma
import tarfile
import zipfile

import pytest
import torch

from fontaine.config.schema import DataConfig, ModelConfig
from fontaine.data import TokenShardDataset, prepare_dataset
from fontaine.data.records import document_from_record, render_conversation
from fontaine.data.sources import iter_documents
from fontaine.inference.chat_template import build_chat_prompt
from fontaine.models import build_model
from fontaine.tokenizer.char_level import CharTokenizer

CHAT = {
    "messages": [
        {"role": "system", "content": "You are Yami."},
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hi there"},
    ]
}
ALPACA = {"instruction": "Add numbers", "input": "2 and 3", "output": "5"}


def _docs(path):
    return list(iter_documents(path))


# -- records ---------------------------------------------------------------------------


def test_chat_records_render_like_the_serving_template():
    rendered = document_from_record(CHAT)
    assert rendered == build_chat_prompt(CHAT["messages"])
    assert rendered.endswith("### Response:\nHi there")


def test_sharegpt_and_bare_message_lists():
    sharegpt = {"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]}
    assert document_from_record(sharegpt) == "### Instruction:\nq1\n\n### Response:\na1"
    assert render_conversation(CHAT["messages"]) == document_from_record(CHAT["messages"])
    assert render_conversation([{"role": "user", "content": "unanswered"}]) is None


def test_instruction_layouts():
    assert document_from_record(ALPACA) == (
        "### Instruction:\nAdd numbers\n\n### Input:\n2 and 3\n\n### Response:\n5"
    )
    assert document_from_record({"question": "why", "answer": "because"}).endswith("because")
    assert document_from_record({"content": "body text"}) == "body text"
    assert document_from_record({"Text": "case-insensitive"}) == "case-insensitive"
    assert document_from_record({"id": 3}) is None


# -- files --------------------------------------------------------------------------------


def test_every_plain_format_is_read(tmp_path):
    (tmp_path / "notes.txt").write_text("para one\n\npara two", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Title\n\nBody", encoding="utf-8")
    (tmp_path / "main.py").write_text("def f():\n\n    return 1\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12", encoding="utf-8")
    (tmp_path / "chat.jsonl").write_text(json.dumps(CHAT) + "\n", encoding="utf-8")
    (tmp_path / "rows.json").write_text(json.dumps({"data": [ALPACA, {"text": "t"}]}), encoding="utf-8")
    with open(tmp_path / "table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instruction", "output"])
        writer.writeheader()
        writer.writerow({"instruction": "csv q", "output": "csv a"})
        writer.writerow({"instruction": "", "output": ""})
    (tmp_path / "table.tsv").write_text("content\nfrom tsv\n", encoding="utf-8")
    notebook = {"cells": [{"cell_type": "markdown", "source": ["# NB"]}, {"cell_type": "code", "source": "x = 1"}]}
    (tmp_path / "nb.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")

    docs = _docs(tmp_path)
    joined = "\n".join(docs)
    assert "para one" in docs and "para two" in docs
    assert "# Title\n\nBody" in docs
    assert "def f():\n\n    return 1" in docs  # code is one document, blank lines kept
    assert "FROM python:3.12" in docs
    assert build_chat_prompt(CHAT["messages"]) in docs
    assert "### Input:\n2 and 3" in joined and "t" in docs
    assert "### Instruction:\ncsv q\n\n### Response:\ncsv a" in docs
    assert "from tsv" in docs
    assert "# NB\n\n```\nx = 1\n```" in docs
    assert "secret" not in joined and "PNG" not in joined


def test_encodings_bom_and_utf16(tmp_path):
    (tmp_path / "bom.txt").write_bytes("\ufeffwith bom".encode())
    (tmp_path / "wide.txt").write_bytes("utf sixteen".encode("utf-16"))
    (tmp_path / "bad.txt").write_bytes(b"ok \xff\xfe? no, only at start" + b"\x80")
    docs = _docs(tmp_path)
    assert "with bom" in docs and "utf sixteen" in docs
    assert any(d.startswith("ok") for d in docs)


def test_csv_without_text_column_is_skipped(tmp_path):
    (tmp_path / "ids.csv").write_text("id,score\n1,2\n", encoding="utf-8")
    assert _docs(tmp_path) == []


# -- archives and compression ----------------------------------------------------------


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_zip_with_mixed_formats_and_nested_archives(tmp_path):
    inner = _zip_bytes({"deep/inner.txt": b"from the inner zip"})
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        payload = b'{"text": "from the tar"}\n'
        info = tarfile.TarInfo("part.jsonl")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive = _zip_bytes(
        {
            "dataset/a.txt": b"zip paragraph",
            "dataset/chat.jsonl": (json.dumps(CHAT) + "\n").encode(),
            "dataset/code/app.js": b"console.log(1)",
            "dataset/more.jsonl.gz": gzip.compress(b'{"text": "gzip inside zip"}\n'),
            "dataset/inner.zip": inner,
            "dataset/bundle.tar.gz": tar_buffer.getvalue(),
            "__MACOSX/dataset/._a.txt": b"junk",
            "dataset/photo.jpg": b"\xff\xd8",
        }
    )
    (tmp_path / "data.zip").write_bytes(archive)
    docs = _docs(tmp_path / "data.zip")
    assert "zip paragraph" in docs
    assert build_chat_prompt(CHAT["messages"]) in docs
    assert "console.log(1)" in docs
    assert "gzip inside zip" in docs
    assert "from the inner zip" in docs
    assert "from the tar" in docs
    assert "junk" not in docs


def test_single_file_compression(tmp_path):
    (tmp_path / "a.txt.gz").write_bytes(gzip.compress(b"gz text"))
    (tmp_path / "b.jsonl.bz2").write_bytes(bz2.compress(b'{"text": "bz2 text"}\n'))
    (tmp_path / "c.md.xz").write_bytes(lzma.compress(b"xz text"))
    assert _docs(tmp_path) == ["gz text", "bz2 text", "xz text"]


def test_tar_archive_on_disk(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "one.txt").write_text("tar one", encoding="utf-8")
    with tarfile.open(tmp_path / "data.tar.xz", "w:xz") as tar:
        tar.add(source / "one.txt", arcname="one.txt")
    assert _docs(tmp_path / "data.tar.xz") == ["tar one"]


def test_broken_archive_is_skipped(tmp_path):
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    assert _docs(tmp_path) == []


def test_parquet_when_pyarrow_is_installed(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"text": ["parquet row"]}), tmp_path / "t.parquet")
    assert _docs(tmp_path) == ["parquet row"]


# -- end to end: zip -> shards -> a modern model learns ------------------------------------


def test_zip_dataset_prepares_and_trains_a_yami_style_model(tmp_path):
    records = [
        {"messages": [{"role": "user", "content": f"count {i}"}, {"role": "assistant", "content": f"{i} {i + 1}"}]}
        for i in range(40)
    ]
    archive = _zip_bytes(
        {
            "chat.jsonl": "\n".join(json.dumps(r) for r in records).encode(),
            "story.md": ("the river runs to the sea. " * 30).encode(),
            "code/util.py": b"def add(a, b):\n    return a + b\n" * 5,
        }
    )
    (tmp_path / "corpus.zip").write_bytes(archive)
    documents = _docs(tmp_path / "corpus.zip")
    tokenizer = CharTokenizer.train(documents)
    config = DataConfig(
        raw_paths=[str(tmp_path / "corpus.zip")],
        output_dir=str(tmp_path / "prepared"),
        sequence_length=32,
        val_fraction=0.0,
        min_document_chars=4,
    )
    result = prepare_dataset(config, tokenizer, name="zip-test")
    assert result.manifest.num_documents == len(documents) == 42
    dataset = TokenShardDataset(result.output_dir, "train", 32)
    assert len(dataset) > 0

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            max_sequence_length=64,
            qk_norm=True,
            sliding_window=16,
            global_attention_every=2,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(min(8, len(dataset)))])
    losses = []
    for _ in range(30):
        loss = model(batch["input_ids"], targets=batch["labels"]).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.6
