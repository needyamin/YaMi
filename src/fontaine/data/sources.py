"""Streaming document readers for raw datasets.

Supported inputs: ``.txt`` (paragraph-split), ``.jsonl`` (one document per
line), ``.json`` (array of strings/objects — loads fully, prefer JSONL for
large files), ``.csv`` (text column). Documents are yielded one at a time so
raw data is never fully loaded into RAM.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from fontaine.utils.logging import get_logger

logger = get_logger("data")

SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".json", ".csv"}
_DEFAULT_TEXT_FIELD = "text"


def _document_from_json_obj(obj: object, source: str) -> str | None:
    """Extract text from a JSON object supporting several common layouts."""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    if _DEFAULT_TEXT_FIELD in obj:
        return str(obj[_DEFAULT_TEXT_FIELD])
    # Instruction/conversation-style records: flattened to prompt + response.
    prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question")
    response = obj.get("response") or obj.get("completion") or obj.get("answer")
    if prompt is not None and response is not None:
        return f"{prompt}\n{response}"
    logger.warning("skipping JSON record without text fields in %s", source)
    return None


def iter_text_file(path: Path) -> Iterator[str]:
    """Yield paragraphs from a text file, splitting on blank lines.

    Streaming: only one paragraph buffer is held in memory at a time, so file
    size is irrelevant to RAM usage.
    """
    buffer: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "":
                if buffer:
                    yield "\n".join(buffer)
                    buffer = []
            else:
                buffer.append(line.rstrip("\n"))
    if buffer:
        yield "\n".join(buffer)


def iter_jsonl_file(path: Path) -> Iterator[str]:
    """Yield one document per JSONL line (the preferred large-scale format)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line %d in %s", line_number, path)
                continue
            document = _document_from_json_obj(obj, str(path))
            if document:
                yield document


def iter_json_file(path: Path) -> Iterator[str]:
    """Yield documents from a JSON array file (loads the file into RAM)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        data = [data]
    for index, obj in enumerate(data):
        document = _document_from_json_obj(obj, f"{path}[{index}]")
        if document:
            yield document


def iter_csv_file(path: Path, text_field: str = _DEFAULT_TEXT_FIELD) -> Iterator[str]:
    """Yield the configured column of a CSV file, streamed row by row."""
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or text_field not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} has no column {text_field!r} "
                f"(columns: {reader.fieldnames})"
            )
        for row in reader:
            value = row.get(text_field)
            if value:
                yield value


def iter_documents(path: str | Path) -> Iterator[str]:
    """Dispatch a file or directory (recursively) to the matching reader."""
    path = Path(path)
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            yield from iter_documents(child)
        return
    if not path.is_file():
        raise FileNotFoundError(f"raw data path not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield from iter_text_file(path)
    elif suffix == ".jsonl":
        yield from iter_jsonl_file(path)
    elif suffix == ".json":
        yield from iter_json_file(path)
    elif suffix == ".csv":
        yield from iter_csv_file(path)
    else:
        logger.warning("skipping unsupported file type: %s (supported: %s)",
                       path, sorted(SUPPORTED_EXTENSIONS))


def iter_source_documents(paths: list[str] | list[Path]) -> Iterator[str]:
    """Yield documents from several raw sources in deterministic order."""
    for entry in paths:
        yield from iter_documents(entry)
