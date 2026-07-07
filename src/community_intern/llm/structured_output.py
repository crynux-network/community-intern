from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_structured_llm_result(
    result: Any,
    response_model: Type[T],
) -> tuple[T, str | None]:
    if isinstance(result, dict) and "parsed" in result:
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            raise RuntimeError("LLM returned unexpected structured output.") from parsing_error
        parsed = _validate_structured_output(result.get("parsed"), response_model)
        return parsed, extract_response_id(result.get("raw"))

    return _validate_structured_output(result, response_model), None


def extract_response_id(raw_response: Any) -> str | None:
    response_metadata = getattr(raw_response, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        return None

    response_id = response_metadata.get("id")
    if not isinstance(response_id, str):
        return None

    response_id = response_id.strip()
    return response_id or None


def _validate_structured_output(value: Any, response_model: Type[T]) -> T:
    if value is None:
        raise RuntimeError("LLM returned null structured output.")

    if isinstance(value, response_model):
        return value

    try:
        return response_model.model_validate(value)
    except Exception as exc:
        expected = response_model.__name__
        actual = type(value).__name__
        raise RuntimeError(
            f"LLM returned unexpected structured output. expected={expected} got={actual}"
        ) from exc
