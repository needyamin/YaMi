"""Chat-message serialization for the Ollama-compatible API.

Ollama-style chat requests arrive as a list of ``{role, content}`` messages;
the engine consumes one plain prompt string. This module joins the two. The
template is Alpaca-style, matching the CodeAlpaca corpus the reference runs
train on (``### Instruction:`` / ``### Response:`` — see
``tools/prepare_codealpaca.py``).
"""

from typing import Any

VALID_ROLES = ("system", "user", "assistant")

# Appended to /api/chat unless the client already sent stop sequences, so the
# model does not continue into a fabricated next user turn.
CHAT_STOP_SEQUENCES = ("\n### Instruction:", "\n### Input:")

# Human-readable description surfaced through POST /api/show.
TEMPLATE_DESCRIPTION = (
    "{{ if .System }}{{ .System }}\n\n{{ end }}"
    "### Instruction:\n{{ .Prompt }}\n\n### Response:\n"
)


class ChatTemplateError(ValueError):
    """Raised when a chat message list cannot be serialized."""


def build_chat_prompt(messages: list[dict[str, Any]]) -> str:
    """Serialize an Ollama-style ``messages`` list into one prompt string.

    Layout (Alpaca-style, same as ``tools/prepare_codealpaca.py``)::

        {system}

        ### Instruction:
        {user}

        ### Response:
        {assistant}

        ### Instruction:
        {user}

        ### Response:

    The prompt ends on the newline after ``### Response:`` so the model writes
    the assistant turn. Prior assistant turns stay inline so multi-turn
    requests
    keep their context.

    ``content`` may be a string or a list of parts (``{"type": "text", "text":
    "..."}``). Image parts are ignored. A message with no remaining text is
    rejected.
    """
    if not isinstance(messages, list) or not messages:
        raise ChatTemplateError("'messages' must be a non-empty list of {role, content}")
    normalized: list[tuple[str, str]] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ChatTemplateError(f"messages[{i}] must be an object with 'role' and 'content'")
        if message.get("role") not in VALID_ROLES:
            raise ChatTemplateError(
                f"messages[{i}].role must be one of {VALID_ROLES}, got {message.get('role')!r}"
            )
        normalized.append((message["role"], _coerce_content(message.get("content"), i)))

    blocks: list[str] = []
    system = "\n\n".join(text for role, text in normalized if role == "system")
    if system:
        blocks.append(system)
    for index, (role, content) in enumerate(normalized):
        if role == "user":
            blocks.append(f"### Instruction:\n{content}\n\n### Response:\n")
        elif role == "assistant":
            if not blocks or not blocks[-1].endswith("### Response:\n"):
                raise ChatTemplateError(
                    f"assistant message at index {index} must follow a user message"
                )
            blocks[-1] += content
    return "\n\n".join(blocks)


def _coerce_content(content: Any, index: int) -> str:
    """Return the text of one message, ignoring non-text parts."""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part.strip())
            elif isinstance(part, dict):
                kind = part.get("type")
                part_text = part.get("text")
                if kind in (None, "text") and isinstance(part_text, str):
                    chunks.append(part_text.strip())
            else:
                raise ChatTemplateError(
                    f"messages[{index}].content parts must be strings or objects"
                )
        text = "\n".join(chunk for chunk in chunks if chunk).strip()
    else:
        raise ChatTemplateError(
            f"messages[{index}].content must be a string or a list of text parts"
        )
    if not text:
        raise ChatTemplateError(f"messages[{index}] has no text content")
    return text
