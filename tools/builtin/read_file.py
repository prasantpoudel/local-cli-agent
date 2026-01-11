from typing import Optional

from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvokation, ToolKind, ToolResult
from utils.path import is_binary_file, resolver_path
from utils.text import count_tokens, truncate_text


class ReadFileParams(BaseModel):
    path: str = Field(
        ...,
        description="Path to the file to read (relative to working directory or absolute)",
    )
    offset: int = Field(
        1, ge=1, description="Line number to start reading from file. Default 1"
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of lines to read. If not specified, reads entire file.",
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Returns the file content with line numbers. "
        "For large files, use offset and limit to read specific portions. "
        "Cannot read binary files (images, executables, etc.)."
    )
    kind = ToolKind.READ
    schema = ReadFileParams

    MAX_FILE_SIZE = 1024 * 1024 * 10  # 10MB
    MAX_OUTPUT_TOKEN = 10000

    async def execute(self, invocation: ToolInvokation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolver_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        if path.is_dir():
            return ToolResult.error_result(f"Given path is a directory path: {path}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(
                f"File too large ({file_size / (1024 * 1024):.1f}MB).",
                f"Maximum is {self.MAX_FILE_SIZE / (1024 * 1024):.0f}MB.",
            )

        if is_binary_file(path):
            file_size_mb = file_size / (1024 * 1024)
            size_str = (
                f"{file_size_mb:.2f}MB" if file_size_mb >= 1 else f"{file_size} bytes"
            )
            return ToolResult.error_result(
                f"Can't read binary file: {path.name} ({size_str})",
                " This tool only read text files.",
            )
        try:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

            lines = content.splitlines()
            total_lines = len(lines)
            if total_lines == 0:
                return ToolResult.success_result(
                    "File is empty.", metadata={"lines": 0}
                )

            start_idx = max(0, params.offset - 1)
            if params.limit:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            selected_lines = lines[start_idx:end_idx]
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=start_idx):
                formatted_lines.append(f"{i:6}|{line}")
            output = "\n".join(formatted_lines)

            token_count = count_tokens(output)
            truncated = False
            if token_count > self.MAX_OUTPUT_TOKEN:
                output = truncate_text(
                    output,
                    self.MAX_OUTPUT_TOKEN,
                    suffix=f"\n... [Truncated {total_lines} total lines]",
                )
                truncated = True

            metadate_line = []
            if start_idx > 0 or end_idx < total_lines:
                metadate_line.append(
                    f"Showing lines {start_idx + 1}-{end_idx} of {total_lines}"
                )

            if metadate_line:
                header = " | ".join(metadate_line) + "\n\n"
                output = header + output

            return ToolResult.success_result(
                output=output,
                truncated=truncated,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "show_start": start_idx + 1,
                    "show_end": end_idx,
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Failes to read file {str(e)}")
