from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass
class TextDelta:
    content: str


@dataclass
class ToolCallDelta:
    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class ToolCall:
    call_id: str
    name: str | None = None
    arguments: str = ""


class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_token: int = 0
    total_token: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_token=self.completion_token + other.completion_token,
            total_token=self.total_token + other.total_token,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class StreamEvent:
    type: StreamEventType
    text_delta: Optional[TextDelta] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_delta: Optional[ToolCallDelta] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


@dataclass
class ToolResultMessage:
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_openai_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


def parse_tool_call_arguments(argument_str: str) -> dict[str, Any]:
    if not argument_str:
        return {}

    try:
        return json.loads(argument_str)
    except json.JSONDecodeError:
        return {"raw_arguments": argument_str}
