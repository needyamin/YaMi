# tools/

Small scripts that sit outside the library:

- `prepare_codealpaca.py` — turn a CodeAlpaca JSON download into Alpaca-template JSONL under `datasets/raw/`.
- `build_blog.py` — build `docs/blog.html` from `docs/book.html`.

Anything that stabilizes gets promoted into `src/fontaine/` with tests.
