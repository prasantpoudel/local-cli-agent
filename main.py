import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from agent.agent import Agent
from agent.event import AgentEventType
from config.loader import Config, load_config
from ui.tui import TUI, get_console
from utils.terminal import get_terminal_logo

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Optional[Agent] = None
        self.tui = TUI(config, console)
        self.config = config

    async def run_single(self, message: str):
        async with Agent(self.config) as agent:
            self.agent = agent

            return await self._process_message(message)

    async def run_interactive(self):
        # console.print(str(get_terminal_logo))
        self.tui.print_welcome(
            "Nova",
            lines=[
                f"model:{self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )
        async with Agent(self.config) as agent:
            self.agent = agent

            while True:
                try:
                    user_input = console.input("\n[user]>[/user] ").strip()
                    if not user_input:
                        continue
                    if user_input.strip() == "/exit":
                        console.print("\n[dim]Goodbye![/dim]")
                        sys.exit(1)
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to quit [/dim]")
                except EOFError:
                    break

    def _get_tool_kind(self, tool_name: str) -> Optional[str]:
        tool = self.agent.tool_registry.get(tool_name)
        if not tool:
            tool_kind = None
        else:
            tool_kind = tool.kind.value
        return tool_kind

    async def _process_message(self, message: str) -> Optional[str]:
        if not self.agent:
            return None
        assistant_streaming = False
        final_response = None
        async for event in self.agent.run(message):
            if event.type == AgentEventType.AGENT_START:
                pass
            elif event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistance_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_streaming:
                    self.tui.end_assistance()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknow error")
                console.print(f"\n[error]Error: {error}[/error]")

            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "Unknwon")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "Unknwon")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_complete(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    success=event.data.get("success", False),
                    output=event.data.get("output", ""),
                    error=event.data.get("error", False),
                    metadata=event.data.get("metadata", None),
                    truncated=event.data.get("truncated", False),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
def main(prompt: Optional[str] = None, cwd: Optional[Path] = None):
    try:
        config: Config = load_config(cwd)
    except Exception as e:
        console.print(f"\n[error]Configuration Error: {e} [/error]")

    errors = config.validate()

    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")

        sys.exit(1)

    cli = CLI(config)
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if not result:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
