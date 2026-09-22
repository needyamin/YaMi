"""Convert the CodeAlpaca-20k JSON dataset into Fontaine-ready JSONL.

Each ``{instruction, input, output}`` record becomes one document rendered in
the classic Alpaca template under a single ``text`` field — the layout the
streaming readers in ``fontaine.data.sources`` pick up directly.

Usage:
    python tools/prepare_codealpaca.py \
        --input datasets/downloads/code_alpaca_20k.json \
        --output datasets/raw/codealpaca_20k.jsonl
"""

import argparse
import json
from pathlib import Path

from fontaine.utils.logging import get_logger

logger = get_logger("tools")

_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
_WITHOUT_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"


def render_document(instruction: str, inp: str, output: str) -> str:
    """Render one Alpaca record; the Input block is omitted when empty."""
    if inp.strip():
        return _WITH_INPUT.format(instruction=instruction, input=inp, output=output)
    return _WITHOUT_INPUT.format(instruction=instruction, output=output)


def convert(input_path: Path, output_path: Path) -> int:
    """Stream records to JSONL and return the number of documents written."""
    with open(input_path, encoding="utf-8") as handle:
        records = json.load(handle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            if not str(record.get("instruction", "")).strip() and not str(
                record.get("output", "")
            ).strip():
                logger.warning("skipping empty record %d", index)
                continue
            text = render_document(
                str(record.get("instruction", "")).strip(),
                str(record.get("input", "")),
                str(record.get("output", "")).strip(),
            )
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CodeAlpaca JSON file")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    args = parser.parse_args()

    written = convert(args.input, args.output)
    size_mb = args.output.stat().st_size / 1e6
    logger.info("wrote %d documents (%.1f MB) to %s", written, size_mb, args.output)


if __name__ == "__main__":
    main()
