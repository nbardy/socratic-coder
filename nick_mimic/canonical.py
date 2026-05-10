from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class UserTurn:
    text: str
    is_sidechain: bool = False
    source_uuid: str | None = None


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    model: str | None = None
    source_uuid: str | None = None


@dataclass(frozen=True)
class ToolCallTurn:
    name: str
    input: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolResultTurn:
    call_id: str | None
    body: str
    is_error: bool = False


@dataclass(frozen=True)
class SystemTurn:
    text: str


Turn = Union[UserTurn, AssistantTurn, ToolCallTurn, ToolResultTurn, SystemTurn]


@dataclass(frozen=True)
class Thread:
    harness: str
    model: str
    cwd: str
    started_at: str | None
    source_path: str
    thread_id: str
    turns: tuple[Turn, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Sample:
    context_messages: list[dict]
    target: str
    project: str
    harness: str
    model: str
    cwd: str
    source_path: str
    thread_id: str
