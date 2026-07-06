from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class LLMSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    api_key: str
    model: str
    vram_limit: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    structured_output_method: Literal["json_schema", "function_calling"] = "function_calling"
    timeout_seconds: float
    max_retries: int

    def chat_crynux_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.vram_limit is not None:
            kwargs["vram_limit"] = self.vram_limit
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        return kwargs
