from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from pydantic import BaseModel, Field


@dataclass(slots=True)
class Turn:
    role: Literal["user", "team", "bot"]
    content: str


@dataclass(slots=True)
class QAPair:
    id: str
    timestamp: str
    turns: list[Turn]
    conversation_id: str = ""
    message_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicEntry:
    filename: str
    description: str


@dataclass(slots=True)
class ClassificationResult:
    topic_name: str


@dataclass(slots=True)
class IntegrationResult:
    skip: bool = False
    remove_ids: list[str] = field(default_factory=list)


class TeamKBState(BaseModel):
    last_processed_qa_id: str = Field(default="")
