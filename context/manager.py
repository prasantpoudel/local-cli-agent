from dataclasses import dataclass, field
from typing import Any, Optional

from config.config import Config
from prompts.system import get_system_prompt
from utils.text import count_tokens


@dataclass
class MessageItem:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result


class ContextManager:
    def __init__(self, config: Config):
        self._system_prompt = get_system_prompt(config)
        self._messages: list[MessageItem] = []
        self._tokenizer_model_name = config.model.name

    def add_user_message(self, content: str) -> None:
        item = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(content, self._tokenizer_model_name),
        )

        self._messages.append(item)

    def add_assistant_message(
        self, content: Optional[str], tool_calls: Optional[list[dict[str, Any]]] = None
    ) -> None:
        # For token counting, we'll just use content for now.
        # In a real app we'd count tool call tokens too.
        count_text = content or ""
        if tool_calls:
            count_text += str(tool_calls)

        item = MessageItem(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            token_count=count_tokens(count_text, self._tokenizer_model_name),
        )

        self._messages.append(item)

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        item = MessageItem(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(content, self._tokenizer_model_name),
        )
        self._messages.append(item)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        for item in self._messages:
            messages.append(item.to_dict())
        return messages
