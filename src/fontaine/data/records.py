"""Turn one structured record (JSON object, CSV row, Parquet row) into a document.

Instruction and chat records are rendered in the Alpaca layout that
``/api/chat`` builds at serve time (``fontaine.inference.chat_template``), so
a model trained on them answers in the same format it is prompted with:

    {system}

    ### Instruction:
    {user}

    ### Input:          (only when an Alpaca ``input`` is present)
    {input}

    ### Response:
    {assistant}

Recognized layouts, checked in this order:

- a plain string, or a ``text`` field;
- chat: ``messages`` / ``conversations`` / ``conversation`` / ``dialog`` lists
  of ``{role, content}`` (OpenAI) or ``{from, value}`` (ShareGPT), or a bare
  list of such messages;
- instruction: ``instruction`` / ``prompt`` / ``question`` / ``query`` plus
  ``output`` / ``response`` / ``completion`` / ``answer`` / ``chosen``, with an
  optional Alpaca ``input`` and ``system``;
- another text field: ``content``, ``body``, ``code``, ``document``, ...
"""

from typing import Any

PROMPT_FIELDS = ("instruction", "prompt", "question", "query", "problem")
RESPONSE_FIELDS = ("output", "response", "completion", "answer", "chosen", "solution", "target")
CHAT_FIELDS = ("messages", "conversations", "conversation", "dialog", "dialogue", "turns")
TEXT_FIELDS = (
    "content",
    "body",
    "code",
    "document",
    "article",
    "story",
    "passage",
    "markdown",
    "raw_content",
    "sentence",
)
SYSTEM_FIELDS = ("system", "system_prompt")

_USER_ROLES = {"user", "human", "prompter", "question", "customer"}
_ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatgpt", "ai", "answer"}
_SYSTEM_ROLES = {"system", "developer"}


def _text(value: Any) -> str:
    """Text of a field that may be a string or a list of ``{type, text}`` parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part.strip())
            elif isinstance(part, dict) and part.get("type") in (None, "text"):
                parts.append(_text(part.get("text")))
        return "\n".join(p for p in parts if p).strip()
    return ""


def render_instruction(instruction: str, output: str, inp: str = "", system: str = "") -> str:
    """One Alpaca turn; the Input block is omitted when empty."""
    blocks = [system.strip()] if system.strip() else []
    turn = f"### Instruction:\n{instruction.strip()}\n\n"
    if inp.strip():
        turn += f"### Input:\n{inp.strip()}\n\n"
    turn += f"### Response:\n{output.strip()}"
    blocks.append(turn)
    return "\n\n".join(blocks)


def _role(message: dict[str, Any]) -> str:
    raw = message.get("role", message.get("from", message.get("speaker", "")))
    role = str(raw).strip().lower()
    if role in _USER_ROLES:
        return "user"
    if role in _ASSISTANT_ROLES:
        return "assistant"
    if role in _SYSTEM_ROLES:
        return "system"
    return ""


def render_conversation(messages: list[Any]) -> str | None:
    """Render chat messages as Alpaca turns; None when no complete turn exists."""
    system_parts: list[str] = []
    turns: list[str] = []
    pending: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _role(message)
        content = _text(message.get("content", message.get("value", message.get("text"))))
        if not role or not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            pending = content if pending is None else f"{pending}\n\n{content}"
        elif pending is not None:
            turns.append(f"### Instruction:\n{pending}\n\n### Response:\n{content}")
            pending = None
        elif turns:
            turns[-1] += f"\n\n{content}"
    if not turns:
        return None
    system = "\n\n".join(system_parts)
    return "\n\n".join(([system] if system else []) + turns)


def _first(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for name in fields:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def document_from_record(record: Any) -> str | None:
    """Best-effort text for one record; None when no known layout matches."""
    if isinstance(record, str):
        return record
    if isinstance(record, list):
        return render_conversation(record)
    if not isinstance(record, dict):
        return None
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    if "text" in lowered and _text(lowered["text"]):
        return _text(lowered["text"])
    chat = _first(lowered, CHAT_FIELDS)
    if isinstance(chat, list):
        return render_conversation(chat)
    prompt = _first(lowered, PROMPT_FIELDS)
    response = _first(lowered, RESPONSE_FIELDS)
    if isinstance(response, list):
        rendered = render_conversation(response)
        if rendered:
            return rendered
    if prompt is not None and response is not None:
        return render_instruction(
            _text(prompt),
            _text(response),
            inp=_text(lowered.get("input") or lowered.get("context")),
            system=_text(_first(lowered, SYSTEM_FIELDS)),
        )
    text = _first(lowered, TEXT_FIELDS)
    if text is not None and _text(text):
        return _text(text)
    return None
