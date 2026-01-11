import logging
from pathlib import Path
from typing import Any

from .base import Tool, ToolInvokation, ToolResult
from .builtin import get_all_builtin_tools

logger = logging.getLogger(__name__)


class ToolRegistery:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tools {tool.name}")

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]
        return None

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for tool in self._tools.values():
            tools.append(tool)
        return tools

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]

    async def invoke(self, name: str, params: dict[str, Any], cwd: Path) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult.error_result(
                error=f"Unknown tool: {name}",
                metadata={"tool_name": name},
            )

        validation_errors = tool.validate_params(params)
        if validation_errors:
            return ToolResult.error_result(
                error=f"Invalid parameters: {';'.join(validation_errors)}",
                metadata={"tool_name": {name}, "errors": {validation_errors}},
            )
        invocation = ToolInvokation(
            params=params,
            cwd=cwd,
        )
        try:
            return await tool.execute(invocation=invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raise unexpected error: {e}")
            return ToolResult.error_result(
                error=f"Internal error {str(e)}", metadata={"tool_name": name}
            )


def create_default_registry() -> ToolRegistery:
    registry = ToolRegistery()
    for tool_class in get_all_builtin_tools():
        registry.register(tool_class())

    return registry
