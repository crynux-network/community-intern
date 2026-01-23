from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from community_intern.core.models import AIResult, Conversation, RequestContext

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class MockAIClient:
    """
    A deterministic AI client for end-to-end adapter testing.

    For any input conversation, returns a fixed reply text.
    """

    reply_text: str = (
        "Mock AI response: thanks for your message. "
        "This is a fixed reply used to test the Discord adapter end-to-end."
    )

    @property
    def project_introduction(self) -> str:
        """Return empty project introduction for testing."""
        return ""

    async def generate_reply(self, conversation: Conversation, context: RequestContext) -> AIResult:
        return AIResult(
            should_reply=True,
            reply_text=self.reply_text,
            debug={
                "mock": True,
                "message_count": len(conversation.messages),
                "platform": context.platform,
            },
        )

    async def invoke_llm(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_model: Type[T],
    ) -> T:
        """Mock LLM invocation that returns dummy responses."""
        _ = system_prompt
        _ = user_content

        try:
            return response_model.model_validate({})
        except Exception:
            return response_model.model_validate(_build_required_placeholders(response_model))


def _build_required_placeholders(model: Type[BaseModel]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        annotation = field.annotation
        if annotation is str:
            data[name] = "mock"
        elif annotation is bool:
            data[name] = False
        elif annotation is int:
            data[name] = 0
        elif annotation is float:
            data[name] = 0.0
        else:
            data[name] = None
    return data
