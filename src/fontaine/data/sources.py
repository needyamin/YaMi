"""Streaming document readers for raw datasets.

``data.raw_paths`` may point at files, directories (walked recursively), or
archives. Every reader works on a stream, so the same formats are read from
disk, from inside ``.zip`` / ``.tar`` archives (nested archives included),
and through ``.gz`` / ``.bz2`` / ``.xz`` / ``.zst`` compression. Nothing is
extracted to disk and documents are yielded one at a time.

| Kind | Extensions | One document is |
| --- | --- | --- |
| plain text | ``.txt``, ``.text`` | a paragraph (blank-line separated) |
| prose / markup | ``.md``, ``.rst``, ``.tex``, ``.html``, ... | the whole file |
| source code | ``.py``, ``.js``, ``.java``, ``.go``, ``.rs``, ... | the whole file |
| records | ``.jsonl``, ``.ndjson``, ``.json``, ``.csv``, ``.tsv``, ``.parquet`` | one record |
| notebooks | ``.ipynb`` | markdown and code cells of one notebook |

Records go through :func:`fontaine.data.records.document_from_record`
(``text``, chat ``messages``, Alpaca ``instruction``/``input``/``output``,
``prompt``/``response``, ...). ``.parquet`` needs ``pyarrow`` and ``.zst``
needs ``zstandard`` (``pip install -e ".[data]"``).
"""

import bz2
import csv
import gzip
import io
import json
import lzma
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import IO, Any

from fontaine.data.records import document_from_record
from fontaine.utils.logging import get_logger

logger = get_logger("data")

PARAGRAPH_EXTENSIONS = {".txt", ".text"}
WHOLE_FILE_EXTENSIONS = {
    # prose and markup
    ".md", ".markdown", ".mdx", ".rst", ".org", ".tex", ".adoc", ".html", ".htm", ".xml", ".srt", ".vtt",
    # source code
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".java", ".kt",
    ".kts", ".scala", ".groovy", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs", ".fs",
    ".go", ".rs", ".swift", ".m", ".mm", ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".jl", ".dart",
    ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".elm", ".zig", ".nim", ".sol", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".sql", ".css", ".scss", ".sass", ".less", ".proto",
    ".graphql", ".tf", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".gradle", ".cmake",
}  # fmt: skip
WHOLE_FILE_NAMES = {"dockerfile", "makefile", "cmakelists.txt", "gemfile", "rakefile", "jenkinsfile"}
RECORD_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet", ".ipynb"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst")
SUPPORTED_EXTENSIONS = (
    PARAGRAPH_EXTENSIONS
    | WHOLE_FILE_EXTENSIONS
    | RECORD_EXTENSIONS
    | set(ARCHIVE_SUFFIXES)
    | set(COMPRESSION_SUFFIXES)
)

# Nested archives are read into memory; this bounds how deep that recursion goes.
MAX_ARCHIVE_DEPTH = 4

Opener = Callable[[], IO[bytes]]


# -- text decoding -------------------------------------------------------------


def _text_stream(binary: IO[bytes], newline: str | None = None) -> io.TextIOWrapper:
    """Decode UTF-8 (with or without BOM) or BOM-marked UTF-16; bad bytes become U+FFFD.

    Line endings are normalized to ``\\n`` unless ``newline=""`` (CSV needs raw ones).
    """
    if not hasattr(binary, "peek"):
        binary = io.BufferedReader(binary)  # type: ignore[arg-type]
    head = binary.peek(4)[:4]  # type: ignore[attr-defined]
    encoding = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return io.TextIOWrapper(binary, encoding=encoding, errors="replace", newline=newline)


# -- per-format readers (text streams) -----------------------------------------


def _paragraphs(handle: IO[str]) -> Iterator[str]:
    buffer: list[str] = []
    for line in handle:
        if line.strip() == "":
            if buffer:
                yield "\n".join(buffer)
                buffer = []
        else:
            buffer.append(line.rstrip("\r\n"))
    if buffer:
        yield "\n".join(buffer)


def _jsonl(handle: IO[str], source: str) -> Iterator[str]:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed JSONL line %d in %s", line_number, source)
            continue
        yield from _emit(obj, f"{source}:{line_number}")


def _json(handle: IO[str], source: str) -> Iterator[str]:
    try:
        data = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("skipping malformed JSON file %s (%s)", source, exc)
        return
    if isinstance(data, dict):
        # Hugging Face / API dumps often wrap the rows: {"data": [...]}.
        for key in ("data", "rows", "records", "examples", "items", "train"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    records = data if isinstance(data, list) else [data]
    if records and all(isinstance(r, dict) and ("role" in r or "from" in r) for r in records):
        records = [records]  # a single conversation stored as a message list
    for index, obj in enumerate(records):
        yield from _emit(obj, f"{source}[{index}]")


def _delimited(handle: IO[str], source: str, delimiter: str) -> Iterator[str]:
    csv.field_size_limit(2**31 - 1)
    reader = csv.DictReader(handle, delimiter=delimiter)
    fields = reader.fieldnames or []
    if not fields:
        return
    single_column = len(fields) == 1
    if not single_column and document_from_record(dict.fromkeys(fields, "x")) is None:
        logger.warning(
            "skipping %s: no text column found (columns: %s; use 'text', 'content', "
            "or instruction/output)",
            source,
            fields,
        )
        return
    for row in reader:
        if single_column:
            document = (row.get(fields[0]) or "").strip()
        else:
            document = document_from_record(row) or ""
        if document:
            yield document


def _notebook(handle: IO[str], source: str) -> Iterator[str]:
    try:
        notebook = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("skipping malformed notebook %s (%s)", source, exc)
        return
    cells = []
    for cell in notebook.get("cells", []) if isinstance(notebook, dict) else []:
        body = cell.get("source", "")
        text = "".join(body) if isinstance(body, list) else str(body)
        if not text.strip():
            continue
        cells.append(f"```\n{text.strip()}\n```" if cell.get("cell_type") == "code" else text.strip())
    if cells:
        yield "\n\n".join(cells)


def _parquet(open_binary: Opener, source: str) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning('skipping %s: reading .parquet needs pyarrow (pip install -e ".[data]")', source)
        return
    with open_binary() as raw:
        seekable = raw if raw.seekable() else io.BytesIO(raw.read())
        parquet = pq.ParquetFile(seekable)
        for batch in parquet.iter_batches(batch_size=1024):
            for index, row in enumerate(batch.to_pylist()):
                yield from _emit(row, f"{source}#{index}")


def _emit(obj: Any, source: str) -> Iterator[str]:
    document = document_from_record(obj)
    if document:
        yield document
    elif document is None:
        logger.warning("skipping record without a text, chat, or instruction layout in %s", source)


# -- dispatch ----------------------------------------------------------------------


def _is_hidden(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return any(part.startswith(".") or part == "__MACOSX" for part in parts if part not in (".", ".."))


def _archive_kind(lower: str) -> str | None:
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(ARCHIVE_SUFFIXES):
        return "tar"
    return None


def _closing_both(outer: Any, inner: IO[bytes]) -> IO[bytes]:
    """Make ``outer.close()`` also close the stream it wraps."""
    close_outer = outer.close

    def close() -> None:
        try:
            close_outer()
        finally:
            inner.close()

    outer.close = close
    return outer


def _decompress(lower: str, raw: IO[bytes]) -> IO[bytes]:
    if lower.endswith(".gz"):
        return _closing_both(gzip.GzipFile(fileobj=raw), raw)
    if lower.endswith(".bz2"):
        return _closing_both(bz2.BZ2File(raw), raw)
    if lower.endswith(".xz"):
        return _closing_both(lzma.LZMAFile(raw), raw)
    try:
        import zstandard
    except ImportError as exc:
        raw.close()
        raise ImportError('reading .zst needs zstandard (pip install -e ".[data]")') from exc
    return zstandard.ZstdDecompressor().stream_reader(raw, closefd=True)  # type: ignore[return-value]


def iter_stream_documents(name: str, open_binary: Opener, depth: int = 0) -> Iterator[str]:
    """Documents from one named byte stream (a file, an archive member, ...).

    ``name`` picks the reader by extension; ``open_binary`` returns a fresh
    readable binary stream each time it is called.
    """
    lower = name.lower()
    base = PurePosixPath(lower.replace("\\", "/")).name
    archive = _archive_kind(lower)
    if archive is not None:
        if depth >= MAX_ARCHIVE_DEPTH:
            logger.warning("skipping %s: archives nested deeper than %d", name, MAX_ARCHIVE_DEPTH)
            return
        reader = _iter_zip if archive == "zip" else _iter_tar
        yield from reader(name, open_binary, depth + 1)
        return
    if lower.endswith(COMPRESSION_SUFFIXES):
        inner = name[: name.rfind(".")]

        def open_inner() -> IO[bytes]:
            return _decompress(lower, open_binary())

        try:
            yield from iter_stream_documents(inner, open_inner, depth)
        except ImportError as exc:
            logger.warning("skipping %s: %s", name, exc)
        return

    suffix = PurePosixPath(base).suffix
    if suffix == ".parquet":
        yield from _parquet(open_binary, name)
        return
    if suffix not in PARAGRAPH_EXTENSIONS | WHOLE_FILE_EXTENSIONS | RECORD_EXTENSIONS and (
        base not in WHOLE_FILE_NAMES
    ):
        logger.warning("skipping unsupported file type: %s", name)
        return
    newline = "" if suffix in (".csv", ".tsv") else None
    with _text_stream(open_binary(), newline) as handle:
        if suffix in PARAGRAPH_EXTENSIONS and base not in WHOLE_FILE_NAMES:
            yield from _paragraphs(handle)
        elif suffix in (".jsonl", ".ndjson"):
            yield from _jsonl(handle, name)
        elif suffix == ".json":
            yield from _json(handle, name)
        elif suffix in (".csv", ".tsv"):
            yield from _delimited(handle, name, "\t" if suffix == ".tsv" else ",")
        elif suffix == ".ipynb":
            yield from _notebook(handle, name)
        else:
            text = handle.read().strip()
            if text:
                yield text


def _seekable_source(open_binary: Opener) -> IO[bytes]:
    raw = open_binary()
    if raw.seekable():
        return raw
    with raw:
        return io.BytesIO(raw.read())


def _iter_zip(name: str, open_binary: Opener, depth: int) -> Iterator[str]:
    with _seekable_source(open_binary) as raw:
        try:
            archive = zipfile.ZipFile(raw)
        except zipfile.BadZipFile as exc:
            logger.warning("skipping %s: not a readable zip file (%s)", name, exc)
            return
        yield from _zip_members(name, archive, depth)


def _zip_members(name: str, archive: zipfile.ZipFile, depth: int) -> Iterator[str]:
    with archive:
        members = sorted(
            (m for m in archive.infolist() if not m.is_dir() and not _is_hidden(m.filename)),
            key=lambda m: m.filename,
        )
        for member in members:
            if member.flag_bits & 0x1:
                logger.warning("skipping encrypted member %s in %s", member.filename, name)
                continue

            def open_member(member: zipfile.ZipInfo = member) -> IO[bytes]:
                return archive.open(member)

            yield from iter_stream_documents(f"{name}/{member.filename}", open_member, depth)


def _iter_tar(name: str, open_binary: Opener, depth: int) -> Iterator[str]:
    with _seekable_source(open_binary) as raw:
        try:
            archive = tarfile.open(fileobj=raw, mode="r:*")
        except tarfile.TarError as exc:
            logger.warning("skipping %s: not a readable tar file (%s)", name, exc)
            return
        yield from _tar_members(name, archive, depth)


def _tar_members(name: str, archive: tarfile.TarFile, depth: int) -> Iterator[str]:
    with archive:
        members = sorted(
            (m for m in archive.getmembers() if m.isfile() and not _is_hidden(m.name)),
            key=lambda m: m.name,
        )
        for member in members:

            def open_member(member: tarfile.TarInfo = member) -> IO[bytes]:
                extracted = archive.extractfile(member)
                if extracted is None:
                    return io.BytesIO(b"")
                return extracted

            yield from iter_stream_documents(f"{name}/{member.name}", open_member, depth)


# -- public entry points --------------------------------------------------------------


def iter_text_file(path: Path) -> Iterator[str]:
    """Yield paragraphs from a text file, splitting on blank lines."""
    with _text_stream(open(path, "rb")) as handle:
        yield from _paragraphs(handle)


def iter_jsonl_file(path: Path) -> Iterator[str]:
    """Yield one document per JSONL line (the preferred large-scale format)."""
    with _text_stream(open(path, "rb")) as handle:
        yield from _jsonl(handle, str(path))


def iter_json_file(path: Path) -> Iterator[str]:
    """Yield documents from a JSON array or wrapped-rows file (loads the file into RAM)."""
    with _text_stream(open(path, "rb")) as handle:
        yield from _json(handle, str(path))


def iter_csv_file(path: Path) -> Iterator[str]:
    """Yield one document per CSV row, streamed."""
    with _text_stream(open(path, "rb"), newline="") as handle:
        yield from _delimited(handle, str(path), ",")


def iter_documents(path: str | Path) -> Iterator[str]:
    """Dispatch a file, archive, or directory (recursively) to the matching reader."""
    path = Path(path)
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            if _is_hidden(str(child.relative_to(path))):
                continue
            yield from iter_documents(child)
        return
    if not path.is_file():
        raise FileNotFoundError(f"raw data path not found: {path}")
    yield from iter_stream_documents(str(path), lambda: open(path, "rb"))


def iter_source_documents(paths: list[str] | list[Path]) -> Iterator[str]:
    """Yield documents from several raw sources in deterministic order."""
    for entry in paths:
        yield from iter_documents(entry)
