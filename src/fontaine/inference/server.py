"""Local inference API and the chat UI.

Three API families share one process:

- Fontaine-native: ``POST /generate`` (JSON + SSE streaming) and ``GET /health``.
- OpenAI-compatible, for the chat UI in ``web/``: ``GET /v1/models`` and
  ``POST /v1/chat/completions`` (SSE).
- Ollama-compatible, so any Ollama client connects to a Fontaine checkpoint
  with no model conversion: ``GET /api/version``, ``GET /api/tags``,
  ``GET /api/ps``, ``POST /api/show``, ``POST /api/chat``,
  ``POST /api/generate``, ``POST /api/context``. Streams are NDJSON.

When ``web/dist`` exists (or ``FONTAINE_WEB_DIST`` points at a built UI),
``GET /`` serves that single-page app. ``GET /api/meta`` is the JSON status.

``POST /api/show`` reports the checkpoint context length and the sampling
defaults the way Ollama does for models such as GLM or DeepSeek
(``*.context_length``, ``PARAMETER num_ctx``, ``capabilities``).
``options.num_ctx`` on a chat request is clamped to that maximum.

Chat is stateless per request: the client sends the full message history and
the server renders it through ``fontaine.inference.chat_template``.

This is intentionally stdlib-only (no FastAPI/uvicorn dependency). The
production path — API gateway, auth, request queue, continuous batching,
GPU workers — is an evolution of this contract, described in
``docs/deployment/serving.md``.
"""

import hashlib
import json
import os
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

# Handled beside sampling overrides. num_ctx selects the context window.
_CONTEXT_OPTION_KEYS = {"num_ctx"}

_STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".map": "application/json",
}


def web_dist_dir() -> Path | None:
    """Directory of the built chat UI, or None when it has not been built."""
    override = os.environ.get("FONTAINE_WEB_DIST", "").strip()
    candidates = [Path(override)] if override else []
    candidates.append(Path.cwd() / "web" / "dist")
    for path in candidates:
        if (path / "index.html").is_file():
            return path.resolve()
    return None


def _is_api_path(path: str) -> bool:
    return path in ("/health", "/generate", "/api", "/v1") or path.startswith(("/api/", "/v1/"))


def _openai_overrides(request: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI chat request into engine sampling overrides."""
    overrides: dict[str, Any] = {}
    if request.get("temperature") is not None:
        overrides["temperature"] = request["temperature"]
    if request.get("top_p") is not None:
        overrides["top_p"] = request["top_p"]
    limit = request.get("max_completion_tokens", request.get("max_tokens"))
    if limit is not None:
        overrides["max_new_tokens"] = limit
    return overrides


def _completion_id() -> str:
    return "chatcmpl-" + hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:24]


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
    unsupported = set(options) - set(_OLLAMA_OPTION_MAP) - _CONTEXT_OPTION_KEYS
    if unsupported:
        logger.debug("ignoring unsupported Ollama options: %s", sorted(unsupported))
    return overrides


def _parse_num_ctx(options: Any) -> int | None:
    """Return a requested context window, or None when the client omitted it."""
    if not isinstance(options, dict) or "num_ctx" not in options:
        return None
    value = options["num_ctx"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("options.num_ctx must be an integer >= 2")
    return value


def _ollama_parameter_text(context_length: int, defaults: dict[str, Any]) -> str:
    """Ollama ``parameters`` blob. Chat clients read this into the model controls."""
    lines = [
        f"num_ctx {context_length}",
        f"temperature {defaults['temperature']}",
        f"top_k {defaults['top_k']}",
        f"top_p {defaults['top_p']}",
        f"repeat_penalty {defaults['repetition_penalty']}",
        f"num_predict {defaults['max_new_tokens']}",
    ]
    return "\n".join(lines) + "\n"


def chat_status_reply(
    model_name: str,
    *,
    error: str | None = None,
    budget: dict[str, object] | None = None,
) -> str:
    """Text to show in the chat bubble when the model writes nothing or fails.

    Chat UIs display assistant message content and skip an empty string, so
    a silent checkpoint otherwise looks like a blank reply.
    """
    if error:
        text = f"{model_name} stopped with an error: {error}"
    else:
        text = (
            f"{model_name} returned no text. "
            "The model stopped before writing a reply. The request reached the model."
        )
    if budget and budget.get("truncated"):
        text += (
            " The prompt was longer than the context window: "
            f"kept {budget['kept_tokens']} of {budget['prompt_tokens']} tokens "
            f"in a window of {budget['context_length']}."
        )
    return text


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
    parameters = generator.parameter_count
    active_parameters = generator.active_parameter_count
    context_length = generator.model.config.max_sequence_length
    architecture = generator.model.config.architecture
    size_bytes = generator.weight_bytes
    digest = "sha256:" + hashlib.sha256(f"{model_name}:{parameters}".encode()).hexdigest()[:12]
    details = {
        "format": "fontaine",
        "family": "Yami",
        "families": ["Yami"],
        "parameter_size": f"{parameters / 1e6:.1f}M",
        "active_parameter_size": f"{active_parameters / 1e6:.1f}M",
        "quantization_level": generator.quantization_level,
    }
    model_info = {
        "general.architecture": architecture,
        "general.parameter_count": parameters,
        "general.active_parameter_count": active_parameters,
        f"{architecture}.context_length": context_length,
    }
    defaults = {
        "temperature": generator.config.temperature,
        "top_k": generator.config.top_k,
        "top_p": generator.config.top_p,
        "repetition_penalty": generator.config.repetition_penalty,
        "max_new_tokens": generator.config.max_new_tokens,
    }
    parameter_text = _ollama_parameter_text(context_length, defaults)
    modelfile = f"# {model_name}\n" + "".join(
        f"PARAMETER {line}\n" for line in parameter_text.strip().splitlines()
    )
    model_entry = {
        "name": model_name,
        "model": model_name,
        "modified_at": _now_stamp(),
        "size": size_bytes,
        "digest": digest,
        "details": details,
    }
    running_entry = {**model_entry, "context_length": context_length}
    web_dist = web_dist_dir()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: N802 (http.server API)
            logger.debug("http: " + format % args)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self) -> str:
            return self.path.split("?", 1)[0]

        def _meta(self) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "status": "ok",
                "model": model_name,
                "parameter_count": parameters,
                "active_parameter_count": active_parameters,
                "context_length": context_length,
                "endpoints": {
                    "openai": ["/v1/models", "/v1/chat/completions"],
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
                "ui": "Chat UI is served at / when web/dist is present.",
            }
            dense_store = getattr(generator, "dense_store", None)
            if dense_store is not None:
                stats = dense_store.stats()
                payload["dense_streaming"] = {
                    "blocks": len(dense_store),
                    "resident_bytes": stats.resident_bytes,
                    "capacity_bytes": stats.capacity_bytes,
                    "requests": stats.requests,
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "prefetched": stats.prefetched,
                    "bytes_read": stats.bytes_read,
                    "directory": str(dense_store.directory),
                }
            return payload

        def do_OPTIONS(self) -> None:  # noqa: N802 (http.server API)
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self._route()
            if path == "/api/meta" or (path == "/" and web_dist is None):
                self._json(200, self._meta())
            elif path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": model_name, "object": "model", "owned_by": "fontaine"}],
                })
            elif path == "/api/version":
                self._json(200, {"version": __version__})
            elif path == "/api/tags":
                self._json(200, {"models": [model_entry]})
            elif path == "/api/ps":
                self._json(200, {"models": [running_entry]})
            elif self._try_static(path):
                return
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
                elif path == "/v1/chat/completions":
                    self._openai_chat(request)
                elif path == "/api/chat":
                    self._ollama_chat(request)
                elif path == "/api/generate":
                    self._ollama_generate(request)
                elif path == "/api/context":
                    self._context_preview(request)
                elif path == "/api/show":
                    self._json(200, {
                        "license": "",
                        "modelfile": modelfile,
                        "parameters": parameter_text,
                        "template": TEMPLATE_DESCRIPTION,
                        "details": details,
                        "model": model_name,
                        "capabilities": ["completion"],
                        "context_length": context_length,
                        "model_info": model_info,
                    })
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                if self._route().startswith("/v1"):
                    self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                else:
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
            self._cors()
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

        def _context_preview(self, request: dict[str, Any]) -> None:
            overrides = _ollama_overrides(request.get("options"))
            context_length_request = _parse_num_ctx(request.get("options"))
            messages = request.get("messages") or []
            prompt = build_chat_prompt(messages) if messages else ""
            self._json(200, generator.context_budget(
                prompt, context_length=context_length_request, **overrides
            ))

        def _ollama_chat(self, request: dict[str, Any]) -> None:
            overrides = _with_chat_stops(_ollama_overrides(request.get("options")))
            context_length_request = _parse_num_ctx(request.get("options"))
            prompt = build_chat_prompt(request.get("messages"))
            if request.get("stream", True):
                self._ndjson_generation(
                    prompt, overrides, mode="chat", context_length=context_length_request
                )
            else:
                chunks = generator.generate(
                    prompt, stream=True, context_length=context_length_request, **overrides
                )
                text = "".join(chunks)
                if not text.strip():
                    text = chat_status_reply(
                        model_name,
                        budget=self._chat_budget(prompt, overrides, context_length_request),
                    )
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
            context_length_request = _parse_num_ctx(request.get("options"))
            if request.get("stream", True):
                self._ndjson_generation(
                    prompt, overrides, mode="generate", context_length=context_length_request
                )
            else:
                chunks = generator.generate(
                    prompt, stream=True, context_length=context_length_request, **overrides
                )
                text = "".join(chunks)
                self._json(200, {
                    "model": model_name,
                    "created_at": _now_stamp(),
                    "response": text,
                    "done_reason": "stop",
                    "done": True,
                })

        def _ndjson_generation(
            self,
            prompt: str,
            overrides: dict,
            mode: str,
            context_length: int | None = None,
        ) -> None:
            """Stream an Ollama-style NDJSON response (one JSON object per line)."""
            self.send_response(200)
            self._cors()
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
            pieces: list[str] = []
            try:
                chunks = generator.generate(
                    prompt, stream=True, context_length=context_length, **overrides
                )
                for delta in chunks:  # type: ignore[union-attr]
                    count += 1
                    pieces.append(delta)
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
                failed: dict[str, Any] = {"error": str(exc), "done": True}
                if mode == "chat":
                    failed["message"] = {
                        "role": "assistant",
                        "content": chat_status_reply(model_name, error=str(exc)),
                    }
                line(failed)
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
                content = ""
                if not "".join(pieces).strip():
                    content = chat_status_reply(
                        model_name,
                        budget=self._chat_budget(prompt, overrides, context_length),
                    )
                final["message"] = {"role": "assistant", "content": content}
            else:
                final["response"] = ""
            line(final)

        def _chat_budget(
            self, prompt: str, overrides: dict, context_length: int | None
        ) -> dict[str, object] | None:
            try:
                return generator.context_budget(
                    prompt, context_length=context_length, **overrides
                )
            except Exception:
                logger.debug("context budget unavailable for the status reply", exc_info=True)
                return None

        def _openai_chat(self, request: dict[str, Any]) -> None:
            overrides = _with_chat_stops(_openai_overrides(request))
            prompt = build_chat_prompt(request.get("messages"))
            if request.get("stream", True):
                self._openai_stream(prompt, overrides)
                return
            chunks = generator.generate(prompt, stream=True, **overrides)
            text = "".join(chunks)
            if not text.strip():
                text = chat_status_reply(
                    model_name, budget=self._chat_budget(prompt, overrides, None)
                )
            self._json(200, {
                "id": _completion_id(),
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })

        def _openai_stream(self, prompt: str, overrides: dict) -> None:
            completion = _completion_id()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(payload: dict[str, Any] | None) -> None:
                if payload is None:
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            def chunk(delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
                return {
                    "id": completion,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            pieces: list[str] = []
            try:
                first = True
                chunks = generator.generate(prompt, stream=True, **overrides)
                for delta in chunks:  # type: ignore[union-attr]
                    pieces.append(delta)
                    body = {"role": "assistant", "content": delta} if first else {"content": delta}
                    first = False
                    emit(chunk(body, None))
                if not "".join(pieces).strip():
                    text = chat_status_reply(
                        model_name, budget=self._chat_budget(prompt, overrides, None)
                    )
                    emit(chunk({"role": "assistant", "content": text}, None))
                emit(chunk({}, "stop"))
                emit(None)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected during OpenAI SSE stream")
            except Exception as exc:
                logger.exception("generation failed")
                try:
                    emit(chunk({"content": chat_status_reply(model_name, error=str(exc))}, "stop"))
                    emit(None)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _try_static(self, path: str) -> bool:
            """Serve the built chat UI. Returns True when this method wrote the response."""
            if web_dist is None:
                return False
            relative = path.lstrip("/")
            if relative:
                candidate = (web_dist / relative).resolve()
                try:
                    candidate.relative_to(web_dist)
                except ValueError:
                    self._json(404, {"error": "not found"})
                    return True
                if candidate.is_file():
                    self._send_file(candidate)
                    return True
            if _is_api_path(path):
                return False
            index = web_dist / "index.html"
            if index.is_file():
                self._send_file(index)
                return True
            return False

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            media = _STATIC_MIME.get(path.suffix.lower(), "application/octet-stream")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
        "OpenAI-compatible API: GET /v1/models | POST /v1/chat/completions"
    )
    logger.info(
        "Ollama-compatible API: POST /api/chat | POST /api/generate | GET /api/tags | POST /api/show"
    )
    logger.info("chat UI: http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server stopped")
    finally:
        server.server_close()
