from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from community_intern.core.formatters import format_message_as_text
from community_intern.core.models import Conversation, Message

_DISCORD_FLOW_TITLE = "Discord AI Flow"
_LLM_FLOW_TITLE = "LLM Flow"
_BOUNDARY_WIDTH = 49
_MAX_TEXT_CHARS = 4000
_BASE64_PATTERN = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=\s]+")


def format_flow_log(
    *,
    title: str = _DISCORD_FLOW_TITLE,
    fields: Sequence[tuple[str, Any]],
) -> str:
    lines = [_opening_boundary(title)]
    for label, value in fields:
        if value is None:
            continue
        text = _format_value(value)
        if "\n" in text:
            lines.append(f"{label}:")
            lines.append(text)
        else:
            lines.append(f"{label}: {text}")
    lines.append("=" * _BOUNDARY_WIDTH)
    return "\n" + "\n".join(lines)


def format_discord_flow_log(*, fields: Sequence[tuple[str, Any]]) -> str:
    return format_flow_log(title=_DISCORD_FLOW_TITLE, fields=fields)


def format_llm_flow_log(*, fields: Sequence[tuple[str, Any]]) -> str:
    return format_flow_log(title=_LLM_FLOW_TITLE, fields=fields)


def format_discord_messages(
    messages: Iterable[Any],
    *,
    role_resolver: Callable[[Any], str] | None = None,
) -> str:
    lines: list[str] = []
    for message in messages:
        text = _normalize_text(getattr(message, "content", "") or "")
        placeholders = _format_discord_attachment_placeholders(message)
        parts = [part for part in [text, *placeholders] if part]
        if not parts:
            continue
        role = role_resolver(message) if role_resolver is not None else "user"
        lines.append(f"{role}: {' | '.join(parts)}")
    return "\n".join(lines) if lines else "No message content."


def format_conversation_messages(conversation_or_messages: Conversation | Sequence[Message]) -> str:
    messages = (
        conversation_or_messages.messages
        if isinstance(conversation_or_messages, Conversation)
        else conversation_or_messages
    )
    lines: list[str] = []
    for message in messages:
        text = _normalize_text("\n".join(format_message_as_text(message)))
        if not text:
            continue
        role = message.role if message.role in {"assistant", "system"} else "user"
        lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else "No message content."


def conversation_log_stats(messages: Sequence[Message]) -> dict[str, int]:
    user_count = sum(1 for message in messages if message.role == "user")
    assistant_count = sum(1 for message in messages if message.role == "assistant")
    image_count = sum(len(message.images or ()) for message in messages)
    attachment_count = sum(len(message.attachments or ()) for message in messages)
    return {
        "message_count": len(messages),
        "user_message_count": user_count,
        "assistant_message_count": assistant_count,
        "message_chars": len(format_conversation_messages(messages)),
        "image_count": image_count,
        "attachment_count": attachment_count,
    }


def text_chars(value: str | None) -> int:
    return len(_normalize_text(value or "", max_chars=None))


def format_text_preview(value: str | None, *, max_chars: int = _MAX_TEXT_CHARS) -> str:
    return _normalize_text(value or "", max_chars=max_chars)


def format_source_ids(source_ids: Iterable[str]) -> str:
    values = [source_id for source_id in source_ids if source_id]
    return ", ".join(values) if values else "None"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "None"
    return "Yes" if value else "No"


def _opening_boundary(title: str) -> str:
    padding = max(_BOUNDARY_WIDTH - len(title) - 2, 0)
    left = padding // 2
    right = padding - left
    return f"{'=' * left} {title} {'=' * right}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return format_bool(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return value if value else "None"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return ", ".join(str(item) for item in value) if value else "None"
    return str(value)


def _format_discord_attachment_placeholders(message: Any) -> list[str]:
    placeholders: list[str] = []
    for attachment in getattr(message, "attachments", []) or []:
        filename = _normalize_text(getattr(attachment, "filename", "") or "")
        content_type = getattr(attachment, "content_type", None) or ""
        is_image = content_type.startswith("image/")
        label = "Image" if is_image else "Attachment"
        placeholders.append(f"{label}: {filename}" if filename else f"{label}: file uploaded")
    return placeholders


def _normalize_text(value: str, *, max_chars: int | None = _MAX_TEXT_CHARS) -> str:
    sanitized = _BASE64_PATTERN.sub("[base64 omitted]", value)
    text = " ".join(line.strip() for line in sanitized.splitlines() if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        return f"{text[:max_chars].rstrip()}... [truncated]"
    return text
