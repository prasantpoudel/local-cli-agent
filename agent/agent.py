from typing import AsyncGenerator

from client.response import StreamEventType, ToolCall, ToolResultMessage
from config.config import Config

from .event import AgentEvent, AgentEventType
from .session import Session


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.session = Session(self.config)

    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        self.session.context_manager.add_user_message(message)
        final_response = None
        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        max_turns = self.config.max_tunrs

        while True and max_turns > self.session._turn_count():
            self.session.increment_turn()
            response_text = ""
            tool_calls: list[ToolCall] = []
            tool_schemas = self.session.tool_registry.get_schemas()

            async for event in self.session.client.chat_completion(
                self.session.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                stream=True,
            ):
                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        yield AgentEvent.text_delta(content=content)
                elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                    if event.tool_calls:
                        tool_calls.append(event.tool_calls)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(event.error or "Unknown error occurs")
                    return

            # Add assistant message to context
            api_tool_calls = None
            if tool_calls:
                api_tool_calls = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            else:
                break

            self.session.context_manager.add_assistant_message(
                response_text or None, tool_calls=api_tool_calls
            )

            tool_call_results: list[ToolResultMessage] = []
            if tool_calls:
                for tool_call in tool_calls:
                    yield AgentEvent.tool_call_start(
                        tool_call.call_id, tool_call.name, tool_call.arguments
                    )

                    result = await self.session.tool_registry.invoke(
                        tool_call.name,
                        tool_call.arguments,
                        self.config.cwd,
                    )
                    yield AgentEvent.tool_call_complete(
                        call_id=tool_call.call_id, name=tool_call.name, result=result
                    )
                    tool_call_results.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.call_id,
                            content=result.to_model_output(),
                            is_error=not result.success,
                        )
                    )

            for tool_result in tool_call_results:
                self.session.context_manager.add_tool_message(
                    tool_result.tool_call_id, tool_result.content
                )

            if response_text:
                yield AgentEvent.text_complete(response_text)

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session and self.session.client:
            await self.session.client.close()
            self.session = None
