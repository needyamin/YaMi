"""Chat-message serialization for the Ollama-compatible API.

Ollama-style chat requests arrive as a list of ``{role, content}`` messages;
the engine consumes one plain prompt string. This module joins the two. The
template is Alpaca-style, matching the CodeAlpaca corpus the reference runs
train on (``### Instruction:`` / ``### Response:``).
"""

from typing import Any

VALID_ROLES = ("system", "user", "assistant")

# Human-readable description surfaced through GET /api/show.
TEMPLATE_DESCRIPTION = (
    "{{ if .System }}{{ .System }}\n\n{{ end }}"
    "### Instruction:\n{{ .Prompt }}\n### Response:\n"
)


class ChatTemplateError(ValueError):
    """Raised when a chat message list cannot be serialized."""


def build_chat_prompt(messages: list[dict[str, Any]]) -> str:
    """Serialize an Ollama-style ``messages`` list into one prompt string.

    Layout (Alpaca-style)::

        {system}

        ### Instruction:
        {user}
        ### Response:
        {assistant}

        ### Instruction:
        {user}
        ### Response:

    The prompt ends at ``### Response:`` so the model writes the assistant
    turn. Prior assistant turns are embedded inline so multi-turn requests
    keep their context.
    """
    if not isinstance(messages, list) or not messages:
        raise ChatTemplateError("'messages' must be a non-empty list of {role, content}")
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ChatTemplateError(f"messages[{i}] must be an object with a string 'content'")
        if message.get("role") not in VALID_ROLES:
            raise ChatTemplateError(
                f"messages[{i}].role must be one of {VALID_ROLES}, got {message.get('role')!r}"
            )

    blocks: list[str] = []
    system = "\n\n".join(m["content"].strip() for m in messages if m["role"] == "system")
    if system:
        blocks.append(system)
    for message in messages:
        content = message["content"].strip()
        if message["role"] == "user":
            blocks.append(f"### Instruction:\n{content}\n### Response:")
        elif message["role"] == "assistant":
            if not blocks or not blocks[-1].endswith("### Response:"):
                raise ChatTemplateError(
                    f"assistant message at index {len(blocks)} must follow a user message"
                )
            blocks[-1] += f" {content}"
    return "\n\n".join(blocks)
