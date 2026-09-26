"""Minimal local inference API (development only).

Two API families share one server:

- Fontaine-native: ``POST /generate`` (JSON + SSE streaming) and
  ``GET /health``.
- Ollama-compatible, so off-the-shelf open-source chat UIs (Open WebUI and
  any other Ollama client) connect to a Fontaine checkpoint directly, with
  no model conversion: ``GET /api/version``, ``GET /api/tags``,
  ``POST /api/show``, ``POST /api/chat``, ``POST /api/generate``. Streams are
  NDJSON (one JSON object per line) per the Ollama API contract.

Chat is stateless per request, like real Ollama: the client sends the full
message history and the server renders it through
``fontaine.inference.chat_template``.

This is intentionally stdlib-only (no FastAPI/uvicorn dependency): it proves
the serving interface and supports local tooling. The production path —
API gateway, auth, request queue, continuous batching, GPU workers — is an
evolution of this contract, described in ``docs/deployment/serving.md``.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fontaine import __version__
from fontaine.inference.chat_template import (
    CHAT_STOP_SEQUENCES,
    TEMPLATE_DESCRIPTION,
    build_chat_prompt,
)
from fontaine.inference.engine import Generator
from fontaine.utils.logging import get_logger

logger = get_logger("inference")

# Product name shown in chat UIs, the same role as "Gemini", "GLM", or "Grok".
DEFAULT_MODEL_NAME = "Yami v1.0"

_OVERRIDABLE_FIELDS = {
    "temperature", "top_k", "top_p", "repetition_penalty", "seed", "max_new_tokens", "stop_sequences",
}

# Ollama request-option name -> Fontaine engine sampling override.
_OLLAMA_OPTION_MAP = {
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "repeat_penalty": "repetition_penalty",
    "num_predict": "max_new_tokens",
    "stop": "stop_sequences",
    "seed": "seed",
}


def infer_model_name(checkpoint: str) -> str:
    """Derive a display name from the checkpoint path (the run folder name).

    Accepts a checkpoints root (``.../<run>/checkpoints``), an exact step
    directory (``.../<run>/checkpoints/step_NNNN``), or any other path.
    """
    path = Path(checkpoint)
    name = path.name
    if name == "checkpoints" or name.startswith("step_"):
        name = path.parent.name
    if name == "checkpoints":  # exact step directory
        name = path.parent.parent.name
    return name or "fontaine"


def _ollama_overrides(options: Any) -> dict[str, Any]:
    """Translate Ollama request ``options`` into engine sampling overrides."""
    if not isinstance(options, dict):
        return {}
    overrides: dict[str, Any] = {}
    for ollama_key, fontaine_key in _OLLAMA_OPTION_MAP.items():
        if ollama_key not in options:
            continue
        value = options[ollama_key]
        if ollama_key == "stop" and isinstance(value, str):
            value = [value]
        overrides[fontaine_key] = value
    unsupported = set(options) - set(_OLLAMA_OPTION_MAP)
    if unsupported:
        logger.debug("ignoring unsupported Ollama options: %s", sorted(unsupported))
    return overrides


def _with_chat_stops(overrides: dict[str, Any]) -> dict[str, Any]:
    """Use the Alpaca turn markers as stops unless the client sent its own."""
    if overrides.get("stop_sequences"):
        return overrides
    return {**overrides, "stop_sequences": list(CHAT_STOP_SEQUENCES)}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_server(
    generator: Generator, host: str, port: int, model_name: str = DEFAULT_MODEL_NAME
) -> ThreadingHTTPServer:
    """Create (but do not start) the development HTTP server."""
    parameters = generator.model.num_parameters()
    active_parameters = generator.model.num_active_parameters()
    size_bytes = sum(p.numel() * p.element_size() for p in generator.model.parameters())
    digest = "sha256:" + hashlib.sha256(f"{model_name}:{parameters}".encode()).hexdigest()[:12]
    details = {
        "format": "fontaine",
        "family": "Yami",
        "families": ["Yami"],
        "parameter_size": f"{parameters / 1e6:.1f}M",
        "active_parameter_size": f"{active_parameters / 1e6:.1f}M",
        "quantization_level": "F32",
    }
    model_entry = {
        "name": model_name,
        "model": model_name,
        "modified_at": _now_stamp(),
        "size": size_bytes,
        "digest": digest,
        "details": details,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: N802 (http.server API)
            logger.debug("http: " + format % args)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self) -> str:
            return self.path.split("?", 1)[0]

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self._route()
            if path in ("/", "/index.html"):
                self._json(200, {
                    "status": "ok",
                    "model": model_name,
                    "parameter_count": parameters,
                    "active_parameter_count": active_parameters,
                    "endpoints": {
                        "ollama": [
                            "/api/version",
                            "/api/tags",
                            "/api/ps",
                            "/api/show",
                            "/api/chat",
                            "/api/generate",
                        ],
                        "native": ["/generate", "/health"],
                    },
                    "ui": "Point an Ollama-compatible UI (e.g. Open WebUI) at this server.",
                })
            elif path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/api/version":
                self._json(200, {"version": __version__})
            elif path == "/api/tags":
                self._json(200, {"models": [model_entry]})
            elif path == "/api/ps":
                self._json(200, {"models": [model_entry]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                path = self._route()
                if path == "/generate":
                    self._native_generate(request)
                elif path == "/api/chat":
                    self._ollama_chat(request)
                elif path == "/api/generate":
                    self._ollama_generate(request)
                elif path == "/api/show":
                    self._json(200, {
                        "license": "",
                        "modelfile": f"# {model_name}\n",
                        "parameters": "",
                        "template": TEMPLATE_DESCRIPTION,
                        "details": details,
                        "model": model_name,
                        "model_info": {
                            "general.architecture": generator.model.config.architecture,
                            "general.parameter_count": parameters,
                            "general.active_parameter_count": active_parameters,
                        },
                    })
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": f"bad request: {exc}"})
            except Exception as exc:  # pragma: no cover - server safety net
                logger.exception("request failed")
                self._json(500, {"error": str(exc)})

        # -- Fontaine-native ---------------------------------------------------

        def _native_generate(self, request: dict[str, Any]) -> None:
            prompt = request.pop("prompt", None)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("'prompt' (non-empty string) is required")
            stream = bool(request.pop("stream", False))
            overrides = {k: request[k] for k in _OVERRIDABLE_FIELDS if k in request}
            if stream:
                self._stream_response(generator, prompt, overrides)
            else:
                text = generator.generate(prompt, **overrides)
                self._json(200, {"text": text})

        def _stream_response(self, generator: Generator, prompt: str, overrides: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(payload: dict[str, Any]) -> None:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            chunks = generator.generate(prompt, stream=True, **overrides)
            try:
                for delta in chunks:  # type: ignore[union-attr]
                    emit({"delta": delta})
                emit({"done": True})
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected during SSE stream")

        # -- Ollama-compatible -------------------------------------------------

        def _ollama_chat(self, request: dict[str, Any]) -> None:
            overrides = _with_chat_stops(_ollama_overrides(request.get("options")))
            prompt = build_chat_prompt(request.get("messages"))
            if request.get("stream", True):
                self._ndjson_generation(prompt, overrides, mode="chat")
            else:
                chunks = generator.generate(prompt, stream=True, **overrides)
                text = "".join(chunks)
                self._json(200, {
                    "model": model_name,
                    "created_at": _now_stamp(),
                    "message": {"role": "assistant", "content": text},
                    "done_reason": "stop",
                    "done": True,
                })

        def _ollama_generate(self, request: dict[str, Any]) -> None:
            prompt = request.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("'prompt' (non-empty string) is required")
            overrides = _ollama_overrides(request.get("options"))
            if request.get("stream", True):
                self._ndjson_generation(prompt, overrides, mode="generate")
            else:
                chunks = generator.generate(prompt, stream=True, **overrides)
                text = "".join(chunks)
                self._json(200, {
                    "model": model_name,
                    "created_at": _now_stamp(),
                    "response": text,
                    "done_reason": "stop",
                    "done": True,
                })

        def _ndjson_generation(self, prompt: str, overrides: dict, mode: str) -> None:
            """Stream an Ollama-style NDJSON response (one JSON object per line)."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            started = time.perf_counter()

            def line(payload: dict[str, Any]) -> None:
                payload.setdefault("model", model_name)
                payload["created_at"] = _now_stamp()
                self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
                self.wfile.flush()

            count = 0
            try:
                chunks = generator.generate(prompt, stream=True, **overrides)
                for delta in chunks:  # type: ignore[union-attr]
                    count += 1
                    if mode == "chat":
                        line({"message": {"role": "assistant", "content": delta}, "done": False})
                    else:
                        line({"response": delta, "done": False})
            except (BrokenPipeError, ConnectionResetError):
                # Client hung up mid-stream (stop button, cancelled request) — not an error.
                logger.debug("client disconnected during NDJSON stream")
                return
            except Exception as exc:  # mid-stream failure: end the stream cleanly
                logger.exception("generation failed")
                line({"error": str(exc), "done": True})
                return
            duration_ns = int((time.perf_counter() - started) * 1e9)
            final: dict[str, Any] = {
                "done_reason": "stop",
                "done": True,
                "total_duration": duration_ns,
                "eval_count": count,
                "eval_duration": duration_ns,
            }
            if mode == "chat":
                final["message"] = {"role": "assistant", "content": ""}
            else:
                final["response"] = ""
            line(final)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(
    generator: Generator, host: str = "127.0.0.1", port: int = 8321, model_name: str = DEFAULT_MODEL_NAME
) -> None:
    """Run the dev server in the foreground (Ctrl+C to stop)."""
    server = build_server(generator, host, port, model_name=model_name)
    logger.info("Fontaine dev server listening on http://%s:%d (model: %s)", host, port, model_name)
    logger.info(
        "Ollama-compatible API: POST /api/chat | POST /api/generate | GET /api/tags | GET /api/version"
    )
    logger.info("connect an Ollama-compatible UI (e.g. Open WebUI) to http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server stopped")
    finally:
        server.server_close()
