"""Minimal local inference API (development only).

Endpoints:
- ``GET  /health``   -> {"status": "ok"}
- ``POST /generate`` -> body: {"prompt": str, "stream": bool, **sampling overrides}

``stream=false`` returns ``{"text": ...}`` once complete; ``stream=true``
sends server-sent events: one ``data: {"delta": "..."}`` line per chunk and a
final ``data: [DONE]``.

This is intentionally stdlib-only (no FastAPI/uvicorn dependency): it proves
the serving interface and supports local tooling. The production path —
API gateway, auth, request queue, continuous batching, GPU workers — is an
evolution of this contract, described in ``docs/deployment/serving.md``.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fontaine.inference.engine import Generator
from fontaine.utils.logging import get_logger

logger = get_logger("inference")

_SAMPLING_FIELDS = {"temperature", "top_k", "top_p", "repetition_penalty", "seed", "stop_sequences"}


def build_server(generator: Generator, host: str, port: int) -> ThreadingHTTPServer:
    """Create (but do not start) the development HTTP server."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence default noise
            logger.debug("http: " + format % args)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                prompt = request.pop("prompt", None)
                if not isinstance(prompt, str) or not prompt:
                    self._json(400, {"error": "'prompt' (non-empty string) is required"})
                    return
                stream = bool(request.pop("stream", False))
                overrides = {k: request[k] for k in _SAMPLING_FIELDS if k in request}
                if stream:
                    self._stream_response(generator, prompt, overrides)
                else:
                    text = generator.generate(prompt, **overrides)
                    self._json(200, {"text": text})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": f"bad request: {exc}"})
            except Exception as exc:  # pragma: no cover - server safety net
                logger.exception("generation failed")
                self._json(500, {"error": str(exc)})

        def _stream_response(self, generator: Generator, prompt: str, overrides: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(payload: dict[str, Any]) -> None:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            chunks = generator.generate(prompt, stream=True, **overrides)
            for delta in chunks:  # type: ignore[union-attr]
                emit({"delta": delta})
            emit({"done": True})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(generator: Generator, host: str = "127.0.0.1", port: int = 8321) -> None:
    """Run the dev server in the foreground (Ctrl+C to stop)."""
    server = build_server(generator, host, port)
    logger.info("Fontaine dev server listening on http://%s:%d", host, port)
    logger.info("POST /generate with {\"prompt\": \"...\"}; GET /health to check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server stopped")
    finally:
        server.server_close()
