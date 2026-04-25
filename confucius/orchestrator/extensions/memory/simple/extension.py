# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from pathlib import Path
from typing import Optional

from IPython.display import Markdown
from pydantic import BaseModel, Field

from .....core import types as cf
from .....core.analect import AnalectRunContext

from .....core.chat_models.bedrock.api.invoke_model import anthropic as ant

from .....utils.artifact import set_artifact
from ....tags import TagLike
from ...file.utils import replace_in_file, view_file_content
from ..reminder import MemoryReminder

from .prompts import (
    EDIT_MEMORY_DESCRIPTION,
    READ_MEMORY_DESCRIPTION,
    SIMPLE_MEMORY_DESCRIPTION,
    WRITE_MEMORY_DESCRIPTION,
)


class ReadMemoryInput(BaseModel):
    # No parameters needed for read
    pass


class WriteMemoryInput(BaseModel):
    content: str = Field(..., description="Full text content to store in memory")


class EditMemoryInput(BaseModel):
    old_str: str = Field(..., description="Exact text to find and replace")
    new_str: str = Field(..., description="Replacement text to insert")


class MemoryOutput(BaseModel):
    content: str = Field(..., description="Memory content or operation result")
    success: bool = Field(..., description="Whether operation succeeded")
    file_path: str = Field(..., description="Path to the memory file")


class SimpleMemoryExtension(MemoryReminder):
    """Simple memory extension that provides persistent text storage."""

    name: str = "simple_memory"
    included_in_system_prompt: bool = True
    memory_namespace: str = Field(
        default="simple_memory", description="Namespace for the memory file"
    )
    memory_dir: str = Field(
        default="/tmp/confucius_memory", description="Directory to store memory files"
    )
    memory_identifier: str = Field(
        default="memory", description="Identifier for the memory file"
    )

    async def description(self) -> TagLike:
        """
        Return a extension description, which will be inserted into the orchestrator system prompt.
        """
        return SIMPLE_MEMORY_DESCRIPTION

    def _get_memory_file_path(self, context: AnalectRunContext) -> Path:
        """Get the path to the memory file for this session."""
        memory_dir = Path(self.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        session_id = context.session or "default"
        return memory_dir / f"{self.memory_namespace}_{session_id}.txt"

    @property
    async def tools(self) -> list[ant.ToolLike]:
        if self.enable_tool_use:
            tools = await super().tools
            return tools + [
                ant.Tool(
                    name="read_memory",
                    description=READ_MEMORY_DESCRIPTION,
                    input_schema=ReadMemoryInput.model_json_schema(),
                ),
                ant.Tool(
                    name="write_memory",
                    description=WRITE_MEMORY_DESCRIPTION,
                    input_schema=WriteMemoryInput.model_json_schema(),
                ),
                ant.Tool(
                    name="edit_memory",
                    description=EDIT_MEMORY_DESCRIPTION,
                    input_schema=EditMemoryInput.model_json_schema(),
                ),
            ]
        return []

    async def _display_memory(self, context: AnalectRunContext) -> None:
        """Display the current contents of memory."""
        self.reset_reminder()
        file_path = self._get_memory_file_path(context)
        if not file_path.exists():
            return

        raw_content = file_path.read_text()
        await set_artifact(
            name=self.memory_identifier,
            value=Markdown(raw_content),
            display_name="Memory",
            collapsed=True,
        )

    async def _read_memory(
        self, inp: ReadMemoryInput, context: AnalectRunContext
    ) -> MemoryOutput:
        """Read memory using file utilities."""
        file_path = self._get_memory_file_path(context)

        await context.io.system(
            f"Reading memory from {file_path}",
            run_label="Reading Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        if not file_path.exists():
            content = "(Memory is empty)"
        else:
            raw_content = file_path.read_text()
            content = view_file_content(
                raw_content,
                start_line=None,
                end_line=None,
                max_view_lines=None,
                include_line_numbers=False,
            )

        await context.io.system(
            "Memory read successfully",
            run_label="Reading Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        return MemoryOutput(content=content, success=True, file_path=str(file_path))

    async def _write_memory(
        self, inp: WriteMemoryInput, context: AnalectRunContext
    ) -> MemoryOutput:
        """Write memory using file utilities."""
        file_path = self._get_memory_file_path(context)

        await context.io.system(
            f"Writing to memory file {file_path}",
            run_label="Writing Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        warning: Optional[str] = None
        if file_path.exists():
            previous_content = file_path.read_text()
            warning = f"Warning: Overwriting existing memory content (previous content had {len(previous_content)} characters)."

        file_path.write_text(inp.content)

        await self._display_memory(context)

        result_msg = "Memory updated successfully."
        if warning:
            result_msg = f"{warning}\n{result_msg}"

        await context.io.system(
            result_msg,
            run_label="Writing Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        return MemoryOutput(content=result_msg, success=True, file_path=str(file_path))

    async def _edit_memory(
        self, inp: EditMemoryInput, context: AnalectRunContext
    ) -> MemoryOutput:
        """Edit memory using file utilities."""
        file_path = self._get_memory_file_path(context)

        await context.io.system(
            f"Editing memory in {file_path}",
            run_label="Editing Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        if not file_path.exists():
            error_msg = "Error: Memory is empty, cannot edit."
            await context.io.system(
                error_msg,
                run_label="Editing Memory",
                run_status=cf.RunStatus.FAILED,
            )
            return MemoryOutput(
                content=error_msg, success=False, file_path=str(file_path)
            )

        try:
            replace_in_file(
                path=file_path,
                find_text=inp.old_str,
                replace_text=inp.new_str,
                require_line_num=False,
            )

            await self._display_memory(context)

            success_msg = f"Memory edited successfully: replaced '{inp.old_str}' with '{inp.new_str}'"
            await context.io.system(
                success_msg,
                run_label="Editing Memory",
                run_status=cf.RunStatus.COMPLETED,
            )

            return MemoryOutput(
                content=success_msg, success=True, file_path=str(file_path)
            )

        except Exception as e:
            error_msg = f"Edit failed: {str(e)}"
            await context.io.system(
                error_msg,
                run_label="Editing Memory",
                run_status=cf.RunStatus.FAILED,
            )
            return MemoryOutput(
                content=error_msg, success=False, file_path=str(file_path)
            )

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Handle tool usage."""

        if tool_use.name == "read_memory":
            inp = ReadMemoryInput.model_validate(tool_use.input)
            result = await self._read_memory(inp, context)

        elif tool_use.name == "write_memory":
            inp = WriteMemoryInput.model_validate(tool_use.input)
            result = await self._write_memory(inp, context)

        elif tool_use.name == "edit_memory":
            inp = EditMemoryInput.model_validate(tool_use.input)
            result = await self._edit_memory(inp, context)

        else:
            # Delegate to parent class for unknown tools (e.g., snooze_reminder)
            return await super().on_tool_use(tool_use, context)

        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content=result.model_dump_json(),
            is_error=not result.success,
        )
