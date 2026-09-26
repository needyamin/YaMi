"""Ollama-compatible API tests: endpoint shapes, streaming format, chat template."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fontaine.config.schema import InferenceConfig
from fontaine.inference import Generator, build_server, infer_model_name
from fontaine.inference.chat_template import ChatTemplateError, build_chat_prompt
from fontaine.models import build_model


@pytest.fixture(scope="module")
def api_base(model_config, tokenizer):
    """A live dev server on an ephemeral port with a tiny greedy model."""
    generator = Generator(
        build_model(model_config),
        tokenizer,
        InferenceConfig(temperature=0.0, max_new_tokens=8),
        device="cpu",
    )
    server = build_server(generator, host="127.0.0.1", port=0, model_name="test-run")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _get(base: str, path: str) -> dict:
    with urlopen(base + path) as resp:
        return json.loads(resp.read())


def _post(base: str, path: str, payload: dict) -> dict:
    request = Request(
        base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlopen(request) as resp:
        return json.loads(resp.read())


def _post_lines(base: str, path: str, payload: dict) -> list[dict]:
    request = Request(
        base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlopen(request) as resp:
        lines = resp.read().decode().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# -- chat template ------------------------------------------------------------


def test_chat_template_single_turn():
    prompt = build_chat_prompt([{"role": "user", "content": "hi"}])
    assert prompt == "### Instruction:\nhi\n\n### Response:\n"


def test_chat_template_system_and_multiturn():
    prompt = build_chat_prompt(
        [
            {"role": "system", "content": "You are Fontaine."},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    )
    assert prompt.startswith("You are Fontaine.")
    assert "### Instruction:\none\n\n### Response:\ntwo" in prompt
    assert prompt.endswith("### Instruction:\nthree\n\n### Response:\n")


def test_chat_template_accepts_text_parts_and_ignores_images():
    prompt = build_chat_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "write a loop"},
                    {"type": "image_url", "image_url": {"url": "http://example/a.png"}},
                ],
            }
        ]
    )
    assert prompt == "### Instruction:\nwrite a loop\n\n### Response:\n"


def test_chat_stops_default_unless_the_client_sent_some():
    from fontaine.inference.server import _with_chat_stops

    assert _with_chat_stops({})["stop_sequences"] == ["\n### Instruction:", "\n### Input:"]
    assert _with_chat_stops({"stop_sequences": ["END"]})["stop_sequences"] == ["END"]


def test_chat_template_rejects_bad_input():
    with pytest.raises(ChatTemplateError):
        build_chat_prompt([])
    with pytest.raises(ChatTemplateError):
        build_chat_prompt([{"role": "tool", "content": "x"}])
    with pytest.raises(ChatTemplateError):
        build_chat_prompt([{"role": "assistant", "content": "orphan"}])


# -- model name derivation ------------------------------------------------------


def test_default_served_name_is_yami_v1(model_config, tokenizer):
    """Chat pickers read /api/tags. The product name is Yami v1.0, not the run folder."""
    assert InferenceConfig().model_name == "Yami v1.0"
    generator = Generator(
        build_model(model_config),
        tokenizer,
        InferenceConfig(temperature=0.0, max_new_tokens=4),
        device="cpu",
    )
    server = build_server(generator, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        models = _get(base, "/api/tags")["models"]
        assert models[0]["name"] == "Yami v1.0"
        assert models[0]["model"] == "Yami v1.0"
        assert models[0]["details"]["family"] == "Yami"
        shown = _post(base, "/api/show", {"name": "Yami v1.0"})
        assert shown["model"] == "Yami v1.0"
        running = _get(base, "/api/ps")["models"]
        assert running[0]["name"] == "Yami v1.0"
    finally:
        server.shutdown()


def test_infer_model_name_variants():
    assert infer_model_name("experiments/2026-01-01_run/checkpoints") == "2026-01-01_run"
    assert (
        infer_model_name("experiments/2026-01-01_run/checkpoints/step_00005000")
        == "2026-01-01_run"
    )
    assert infer_model_name("some/model.bin") == "model.bin"
    assert infer_model_name("") == "fontaine"


# -- endpoints ------------------------------------------------------------------


def test_version_and_tags(api_base):
    assert "version" in _get(api_base, "/api/version")
    models = _get(api_base, "/api/tags")["models"]
    assert len(models) == 1
    assert models[0]["name"] == "test-run"
    assert models[0]["details"]["parameter_size"].endswith("M")


def test_root_status_json(api_base):
    status = _get(api_base, "/")
    assert status["model"] == "test-run"
    assert "/api/chat" in status["endpoints"]["ollama"]
    assert "/api/ps" in status["endpoints"]["ollama"]
    assert status["active_parameter_count"] == status["parameter_count"]
    assert status["active_parameter_count"] > 0


def test_running_models_endpoint(api_base):
    models = _get(api_base, "/api/ps")["models"]
    assert len(models) == 1
    assert models[0]["name"] == "test-run"
    assert models[0]["details"]["active_parameter_size"].endswith("M")


def test_chat_non_streaming(api_base):
    response = _post(
        api_base,
        "/api/chat",
        {"messages": [{"role": "user", "content": "the fontaine"}], "stream": False},
    )
    assert response["done"] is True
    assert response["message"]["role"] == "assistant"
    # The untrained toy model may emit EOS immediately; only the shape is asserted.
    assert isinstance(response["message"]["content"], str)


def test_chat_streaming_ndjson_shape(api_base):
    lines = _post_lines(
        api_base,
        "/api/chat",
        {"messages": [{"role": "user", "content": "the fontaine"}], "options": {"num_predict": 8}},
    )
    assert len(lines) >= 1
    for line in lines[:-1]:
        assert line["done"] is False
        assert line["message"]["role"] == "assistant"
        assert isinstance(line["message"]["content"], str)
    final = lines[-1]
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["eval_count"] == len(lines) - 1
    assert final["message"]["content"] == ""


def test_generate_endpoint_non_streaming_and_streaming(api_base):
    full = _post(api_base, "/api/generate", {"prompt": "the", "stream": False})
    assert full["done"] is True and isinstance(full["response"], str) and full["response"]

    lines = _post_lines(api_base, "/api/generate", {"prompt": "the", "stream": True})
    assert len(lines) >= 2  # the toy model greedily continues raw prompts
    assert all(line["done"] is False and line["response"] for line in lines[:-1])
    assert lines[-1]["done"] is True and lines[-1]["response"] == ""


def test_chat_rejects_bad_messages(api_base):
    with pytest.raises(HTTPError) as excinfo:
        _post(api_base, "/api/chat", {"messages": []})
    assert excinfo.value.code == 400
    with pytest.raises(HTTPError) as excinfo:
        _post(
            api_base,
            "/api/chat",
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
        )
    assert excinfo.value.code == 400


def test_chat_accepts_list_content(api_base):
    response = _post(
        api_base,
        "/api/chat",
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "the fontaine"}]}
            ],
            "stream": False,
        },
    )
    assert response["done"] is True
    assert isinstance(response["message"]["content"], str)


def test_native_generate_contract_unchanged(api_base):
    response = _post(api_base, "/generate", {"prompt": "the fontaine"})
    assert isinstance(response["text"], str) and response["text"]
    assert _get(api_base, "/health") == {"status": "ok"}
