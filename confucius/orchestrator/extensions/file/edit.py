# pyre-strict

import asyncio
import difflib
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Callable, cast, Literal

import bs4
from pydantic import Field, model_validator, validator
from typing_extensions import Self

from ....core import types as cf

from ....core.analect.analect import AnalectRunContext, get_current_context
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ....core.memory import CfMessage
from ....utils.pydantic import sanitize_pydantic_validation_error
from ....utils.string import truncate
from ...exceptions import OrchestratorInterruption
from ...tags import Tag, TagLike
from ..function.utils import Function, get_runnable
from ..tag_with_id import TagWithIDExtension, unescape
from ..tool_use import ToolUseExtension
from .policy.base import FileAccessPolicyBase
from .policy.open import OpenFileAccessPolicy


# Minimal change type enum (added/modified/removed)
class ChangeType(str, Enum):
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    REMOVED = "REMOVED"


from .history import FileHistory
from .prompts import (
    FILE_EDIT_BASIC_PROMPT,
    FILE_EDIT_TOOL_USE_USER_ERR_MESSAGE,
    FILE_EDIT_TOOL_USER_ACCESS_DENIED_MESSAGE,
)
from .utils import (
    create_file,
    delete_file,
    escape_file_content,
    insert_in_file,
    replace_in_file,
    view_directory,
    view_file,
)

FileCallback = Callable[[Path], Awaitable[str | None] | str | None]
HISTORY_KEY = "__file_edit_history__"
LAST_ERROR_TYPE_KEY = "__file_edit_last_error_type__"


async def test_no_side_effects_callback(path: Path) -> str:
    """Test callback with no side effects - just returns path info for demonstration purposes."""
    return f"No-side-effects callback processed: {path.name}"


class FileAccessError(Exception):
    """Raised when an operation is not allowed."""

    pass


class FileEditExtension(TagWithIDExtension, ToolUseExtension):
    tag_name: str = "file_edit"
    default_identifier: str = "file_edit"
    trace_tool_execution: bool = False  # Custom trace node will be generated
    max_output_lines: int = Field(
        default=300,
        description="Maximum number of lines of the output to include in the response string",
    )
    on_file_changed: list[FileCallback] = Field(
        default=[],
        description="Callback function to be called when a file is changed, this function must return a string, which will be used as feedbacks to the LLM",
    )
    on_file_changed_no_side_effects: list[FileCallback] = Field(
        default=[],
        description="Callback functions with no side effects to be called when a file is changed. These functions only read/analyze and do not modify files or system state.",
    )
    on_file_changed_with_side_effects: list[FileCallback] = Field(
        default=[],
        description="Callback functions with side effects to be called when a file is changed. These functions may modify files, run formatters, linters, or change system state.",
    )
    display_diff: bool = Field(
        default=True,
        description="Whether to display the diff of the changes through the IO interface",
    )
    display_diff_context_lines: int = Field(
        default=5,
        description="Number of context lines to display in the diff, only used when display_diff is True",
    )
    display_view: bool = Field(
        default=True,
        description="Whether to display the viewing of the content of the file through the IO interface",
    )
    enable_tool_use: bool = False
    editor_tool: ant.TextEditor = Field(
        default_factory=ant.TextEditor,
        description="The text editor to use for editing the file, only needed when enable_tool_use is True",
    )
    access_policy: FileAccessPolicyBase = Field(
        default_factory=OpenFileAccessPolicy,
        description="Policy controlling file access permissions",
    )

    @validator("on_file_changed", pre=True, always=True)
    def _validate_on_file_changed(
        cls,  # noqa: B902
        v: list[FileCallback] | FileCallback,
    ) -> list[FileCallback]:
        if isinstance(v, list):
            return v
        return [v]

    @validator("on_file_changed_no_side_effects", pre=True, always=True)
    def _validate_on_file_changed_no_side_effects(
        cls,  # noqa: B902
        v: list[FileCallback] | FileCallback,
    ) -> list[FileCallback]:
        if isinstance(v, list):
            return v
        return [v]

    @validator("on_file_changed_with_side_effects", pre=True, always=True)
    def _validate_on_file_changed_with_side_effects(
        cls,  # noqa: B902
        v: list[FileCallback] | FileCallback,
    ) -> list[FileCallback]:
        if isinstance(v, list):
            return v
        return [v]

    # pyre-fixme[56]: Invalid decoration [56]
    @model_validator(mode="after")
    def _validate_tool_use(self) -> Self:  # noqa: B902
        if self.enable_tool_use:
            self.included_in_system_prompt = False
        return self

    def _categorize_callbacks(
        self, callbacks: list[FileCallback]
    ) -> tuple[list[FileCallback], list[FileCallback]]:
        """Categorize callbacks based on their side effect metadata."""
        no_side_effects = [
            cb
            for cb in callbacks
            if hasattr(cb, "has_side_effects")
            and not getattr(cb, "has_side_effects", True)
        ]
        with_side_effects = [
            cb
            for cb in callbacks
            if not hasattr(cb, "has_side_effects")
            or getattr(cb, "has_side_effects", True)
        ]
        return no_side_effects, with_side_effects

    # pyre-fixme[56]: Invalid decoration [56]
    @model_validator(mode="after")
    def _auto_categorize_callbacks(self) -> Self:  # noqa: B902
        """Automatically categorize callbacks when legacy on_file_changed is used."""
        if (
            self.on_file_changed
            and not self.on_file_changed_no_side_effects
            and not self.on_file_changed_with_side_effects
        ):
            # Legacy usage - auto-categorize the callbacks
            no_effects, with_effects = self._categorize_callbacks(self.on_file_changed)
            object.__setattr__(self, "on_file_changed_no_side_effects", no_effects)
            object.__setattr__(self, "on_file_changed_with_side_effects", with_effects)
        return self

    def get_all_callbacks(self) -> list[FileCallback]:
        """Get all callbacks combined from both categorized lists."""
        return (
            self.on_file_changed_no_side_effects
            + self.on_file_changed_with_side_effects
        )

    @property
    def _history(self) -> FileHistory:
        """
        Use a consistent history object for the entire session
        """
        context = get_current_context()
        return cast(
            FileHistory,
            context.session_storage[self.__class__.__name__].setdefault(
                HISTORY_KEY, FileHistory()
            ),
        )

    def _get_session_storage(self, context: AnalectRunContext) -> dict[str, Any]:
        """
        Get the session storage namespace for this extension.
        """
        return context.session_storage[self.__class__.__name__]

    async def on_llm_output(
        self,
        text: str,
        context: AnalectRunContext,
    ) -> str:
        if self.enable_tool_use:
            return text

        return escape_file_content(text)

    async def description(self) -> TagLike:
        return FILE_EDIT_BASIC_PROMPT.format(file_edit_tag_name=self.tag_name)

    async def _check_policy(
        self,
        operation: Literal["read", "create", "update", "delete"],
        path: Path,
        context: AnalectRunContext,
        is_directory: bool = False,
    ) -> None:
        """Check if an operation is allowed by the policy.

        Args:
            operation: The operation type to check ("read", "create", "update", "delete")
            path: The path to check
            context: The run context
            is_directory: Whether the path is a directory (only for read operations)

        Returns:
            The error message if operation is denied, None if allowed
        """
        method_name = f"check_{operation}"
        policy_method = getattr(self.access_policy, method_name)

        # Handle is_directory parameter for read operation
        kwargs = {}
        if operation == "read":
            kwargs["is_directory"] = is_directory

        result = await policy_method(path, context=context, **kwargs)
        if not result.allowed:
            # Generate appropriate default error message based on operation
            default_message = (
                f"I don't have permission to {operation} the file at {path}"
            )
            message = result.message or default_message

            raise FileAccessError(message)

    async def _run_func(
        self,
        func: Function,
        context: AnalectRunContext,
        fallback_name: str | None = None,
        /,
        **kwargs: Any,
    ) -> object:
        """
        Run a function with the given arguments and return the result.

        Args:
            func (Function): The function to run.
            context (AnalectRunContext): The context of the run.
            fallback_name (str, optional): The name of the function to use if the function is not specified. Defaults to None.

        Returns:
            object: The result of the function.
        """

        runnable = get_runnable(func)
        if not hasattr(runnable, "name") or not runnable.name:
            runnable.name = fallback_name
        return await context.invoke(runnable, kwargs, run_type="tool")

    async def _display_view(
        self,
        file_path: Path,
        start_line: int | None,
        end_line: int | None,
        context: AnalectRunContext,
    ) -> None:
        # Implementation to be customized by subclasses
        return None

    async def _display_diff(
        self,
        path: Path,
        context: AnalectRunContext,
        *,
        change_type: ChangeType,
        before_text: str | None = None,
    ) -> str:
        """Display the diff of the changes through the IO interface, and return unified_diff patch string"""

        before_text = (
            _read_text_from_path(self._history.get_last_version(path))
            if before_text is None
            else before_text
        )
        after_text = _read_text_from_path(path)

        if before_text == after_text:
            return ""

        return "\n".join(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=str(path),
                tofile=str(path),
            )
        )

    async def _on_file_changed(
        self,
        path: Path,
        context: AnalectRunContext,
        *,
        change_type: ChangeType,
        before_text: str | None = None,
    ) -> str:
        """Run the on_file_changed callback function and return the result as a string

        Args:
            path (Path): The path of the file that was changed.
            context (AnalectRunContext): The context of the run.
            change_type (ChangeType): The type of change that was made to the file.
            before_text (str, optional): The text of the file before the change. If not provided, it will be read from the history. (Will be used in undo operation)

        Returns:
            str: The result of the callback function as a string. If the callback function returns None, an empty string will be returned.
        """

        # Check if there are any callbacks to run
        all_callbacks = self.get_all_callbacks()
        results = []
        if all_callbacks:
            await context.io.system(
                f"Running callbacks to process file changes at `{path.name}`",
                run_label="Processing file",
                run_status=cf.RunStatus.IN_PROGRESS,
            )

        # First, run callbacks with side effects serially
        for idx, func in enumerate(self.on_file_changed_with_side_effects):
            result = await self._run_func(
                func, context, f"on_file_changed_with_side_effects_{idx}", path=path
            )
            if result is not None:
                results.append(result)

        # Then, run callbacks with no side effects in parallel
        if self.on_file_changed_no_side_effects:
            no_side_effects_tasks = [
                self._run_func(
                    func, context, f"on_file_changed_no_side_effects_{idx}", path=path
                )
                for idx, func in enumerate(self.on_file_changed_no_side_effects)
            ]
            no_side_effects_results = await asyncio.gather(
                *no_side_effects_tasks, return_exceptions=True
            )

            # Process results, filtering out exceptions and None values
            for result in no_side_effects_results:
                if not isinstance(result, Exception) and result is not None:
                    results.append(result)

        if self.display_diff:
            try:
                patch = await self._display_diff(
                    path, context, change_type=change_type, before_text=before_text
                )
                if patch:
                    results.append(
                        dedent(
                            """\
                        Here is the diff of the changes after your edits and all the registered callbacks:
                        
                        ```console
                        {patch}
                        ```
                        """
                        ).format(patch=patch)
                    )
            except Exception as e:
                await context.io.system(
                    f"Failed to display diff due to {type(e).__name__}: {str(e)}",
                    run_label="Display diff failed",
                    run_status=cf.RunStatus.FAILED,
                )

        if not results:
            return ""
        else:
            return "\n" + "\n".join(results)

    async def _on_create(
        self, file_path: Path, content: str, context: AnalectRunContext, **kwargs: Any
    ) -> str:
        # Check create permissions first
        await self._check_policy("create", file_path, context)

        await context.io.system(
            f"Creating file at `{file_path.name}`", run_label="Creating file"
        )
        self._history.save_state(file_path)
        await self._run_func(
            create_file, context, path=file_path, content=content, **kwargs
        )
        msg = f"File created successfully at `{file_path.name}`" + (
            await self._on_file_changed(
                file_path, context, change_type=ChangeType.ADDED
            )
        )
        await context.io.system(
            msg, run_label="File created", run_status=cf.RunStatus.COMPLETED
        )
        return msg

    async def _on_create_tag(
        self, file_path: Path | None, tag: bs4.Tag, context: AnalectRunContext
    ) -> str:
        if not file_path:
            raise ValueError('"file_path" is required for create operation')

        # Check create permissions
        await self._check_policy("create", file_path, context)

        content = unescape(tag.decode_contents().strip("\n"))
        return await self._on_create(
            file_path=file_path, content=content, context=context
        )

    async def _on_replace(
        self,
        file_path: Path,
        find_text: str,
        replace_text: str,
        context: AnalectRunContext,
        **kwargs: Any,
    ) -> str:
        # Check update permissions first
        await self._check_policy("update", file_path, context)

        await context.io.system(
            f"Replacing content in file at `{file_path.name}`",
            run_label="Replacing content",
        )
        self._history.save_state(file_path)
        await self._run_func(
            replace_in_file,
            context,
            path=file_path,
            find_text=find_text,
            replace_text=replace_text,
            **kwargs,
        )
        msg = f"File replaced successfully at `{file_path.name}`" + (
            await self._on_file_changed(
                file_path, context, change_type=ChangeType.MODIFIED
            )
        )

        await context.io.system(
            msg, run_label="File replaced", run_status=cf.RunStatus.COMPLETED
        )
        return msg

    async def _on_replace_tag(
        self,
        file_path: Path | None,
        find_tag: bs4.Tag | None,
        replace_tag: bs4.Tag | None,
        context: AnalectRunContext,
    ) -> str:
        if not file_path:
            raise ValueError('"file_path" is required for replace operation')
        if not find_tag or not replace_tag:
            raise ValueError(
                "Both <find> and <replace> tags are required for replace operation"
            )

        # Check update permissions
        await self._check_policy("update", file_path, context)

        find_text = unescape(find_tag.decode_contents().strip("\n"))
        replace_text = unescape(replace_tag.decode_contents().strip("\n"))
        return await self._on_replace(
            file_path=file_path,
            find_text=find_text,
            replace_text=replace_text,
            context=context,
        )

    async def _on_insert(
        self,
        file_path: Path,
        find_after: str | None,
        content: str,
        context: AnalectRunContext,
        **kwargs: Any,
    ) -> str:
        # Check update permissions first
        await self._check_policy("update", file_path, context)

        await context.io.system(
            f"Inserting content in file at `{file_path.name}`",
            run_label="Inserting content",
        )
        self._history.save_state(file_path)
        await self._run_func(
            insert_in_file,
            context,
            path=file_path,
            find_after=find_after,
            content=content,
            **kwargs,
        )
        msg = f"File inserted successfully at `{file_path.name}`" + (
            await self._on_file_changed(
                file_path, context, change_type=ChangeType.MODIFIED
            )
        )

        await context.io.system(
            msg, run_label="File inserted", run_status=cf.RunStatus.COMPLETED
        )
        return msg

    async def _on_insert_tag(
        self,
        file_path: Path | None,
        find_after_tag: bs4.Tag | None,
        content_tag: bs4.Tag | None,
        context: AnalectRunContext,
    ) -> str:
        if not file_path:
            raise ValueError('"file_path" is required for insert operation')
        if not find_after_tag or not content_tag:
            raise ValueError("Both `<find_after>` and `<content>` tags are required")

        # Check update permissions
        await self._check_policy("update", file_path, context)

        find_after = unescape(find_after_tag.decode_contents().strip("\n"))
        content = unescape(content_tag.decode_contents().strip("\n"))
        return await self._on_insert(
            file_path=file_path,
            find_after=find_after,
            content=content,
            context=context,
        )

    async def _on_delete(
        self,
        file_path: Path | None,
        context: AnalectRunContext,
    ) -> str:
        if not file_path:
            raise ValueError("file_path is required for delete operation")

        # Check delete permissions first
        await self._check_policy("delete", file_path, context)

        # Delete entire file
        await context.io.system(
            f"Deleting file at `{file_path.name}`", run_label="Deleting file"
        )
        self._history.save_state(file_path)
        await self._run_func(delete_file, context, path=file_path)
        msg = f"File deleted successfully at `{file_path.name}`" + (
            await self._on_file_changed(
                file_path, context, change_type=ChangeType.REMOVED
            )
        )

        await context.io.system(
            msg, run_label="File deleted", run_status=cf.RunStatus.COMPLETED
        )
        return msg

    async def _on_view_file_path(
        self,
        file_path: Path,
        start_line: int | None,
        end_line: int | None,
        context: AnalectRunContext,
    ) -> str:
        # Check read permissions first
        await self._check_policy("read", file_path, context)

        await context.io.system(
            f"Viewing file at `{file_path.name}`", run_label="Viewing file"
        )
        res = await self._run_func(
            view_file,
            context,
            path=file_path,
            start_line=start_line,
            end_line=end_line,
            max_view_lines=self.max_output_lines,
        )
        await context.io.system(
            f"File content viewed at `{file_path.name}`, lines: {start_line} - {end_line}",
            run_label="File viewed",
            run_status=cf.RunStatus.COMPLETED,
        )
        if self.display_view:
            try:
                await self._display_view(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    context=context,
                )
            except Exception as e:
                await context.io.system(
                    f"Failed to display file content due to {type(e).__name__}: {str(e)}",
                    run_label="Display file content failed",
                    run_status=cf.RunStatus.FAILED,
                )

        attributes = {
            "file_path": str(file_path),
        }
        if start_line:
            attributes["start_line"] = str(start_line)
        if end_line:
            attributes["end_line"] = str(end_line)

        return Tag(name="view", attributes=attributes, contents=str(res)).prettify()

    async def _on_view_directory(
        self,
        directory: Path,
        depth: int,
        show_hidden: bool,
        context: AnalectRunContext,
    ) -> str:
        # Check directory read permissions first
        await self._check_policy("read", directory, context, is_directory=True)

        await context.io.system(
            f"Viewing directory at `{directory.name}`",
            run_label="Viewing directory",
        )
        res = await self._run_func(
            view_directory,
            context,
            path=directory,
            depth=depth,
            show_hidden=show_hidden,
        )
        await context.io.system(
            f"Directory content viewed at `{directory.name}`, depth: {depth}, show_hidden: {show_hidden}",
            run_label="Directory viewed",
            run_status=cf.RunStatus.COMPLETED,
        )
        attributes = {
            "directory": str(directory),
            "depth": str(depth),
        }
        if show_hidden:
            attributes["show_hidden"] = "true"
        return (
            Tag(
                name="view",
                attributes=attributes,
                contents=truncate(str(res), max_lines=self.max_output_lines),
            )
            .prettify()
            .strip()
        )

    async def _on_view_tag(
        self,
        file_path: Path | None,
        directory: Path | None,
        tag: bs4.Tag,
        context: AnalectRunContext,
    ) -> str:
        if not file_path and not directory:
            raise ValueError(
                "Either file_path or directory is required for view operation"
            )
        if file_path:
            start_line = tag.get("start_line")
            end_line = tag.get("end_line")
            start_line = int(start_line) if start_line else None
            end_line = int(end_line) if end_line else None
            return await self._on_view_file_path(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                context=context,
            )
        else:
            assert directory is not None, "directory must be set"
            depth = int(tag.get("depth", 2))
            show_hidden = "show_hidden" in tag.attrs
            return await self._on_view_directory(
                directory=directory,
                depth=depth,
                show_hidden=show_hidden,
                context=context,
            )

    async def _on_undo(
        self,
        file_path: Path | None,
        context: AnalectRunContext,
    ) -> str:
        if not file_path:
            raise ValueError("file_path is required for undo operation")

        await context.io.system(
            f"Undoing last change to file at `{file_path}`", run_label="Undoing changes"
        )

        before_text = _read_text_from_path(file_path)
        self._history.undo(file_path)
        msg = f"Successfully undid last change to file at `{file_path}`" + (
            await self._on_file_changed(
                file_path,
                context,
                change_type=ChangeType.MODIFIED,
                before_text=before_text,
            )
        )

        await context.io.system(
            msg, run_label="Changes undone", run_status=cf.RunStatus.COMPLETED
        )
        return msg

    async def _on_file_edit_tag(
        self,
        tag: bs4.Tag,
        context: AnalectRunContext,
    ) -> None:
        # Get common attributes
        file_path = Path(tag.get("file_path")) if tag.get("file_path") else None
        directory = Path(tag.get("directory")) if tag.get("directory") else None

        op_type = tag.get("type")
        msg = None
        try:
            if op_type == "create":
                msg = await self._on_create_tag(
                    file_path=file_path, tag=tag, context=context
                )
            elif op_type == "replace":
                msg = await self._on_replace_tag(
                    file_path=file_path,
                    find_tag=tag.find("find"),
                    replace_tag=tag.find("replace"),
                    context=context,
                )
            elif op_type == "insert":
                msg = await self._on_insert_tag(
                    file_path=file_path,
                    find_after_tag=tag.find("find_after"),
                    content_tag=tag.find("content"),
                    context=context,
                )
            elif op_type == "delete":
                msg = await self._on_delete(
                    file_path=file_path,
                    context=context,
                )
            elif op_type == "view":
                msg = await self._on_view_tag(
                    file_path=file_path,
                    directory=directory,
                    tag=tag,
                    context=context,
                )
            elif op_type == "undo":
                msg = await self._on_undo(file_path=file_path, context=context)
            elif op_type is None:
                raise ValueError(
                    "Operation type must be specified with 'type' attribute, do not write <file_edit> tag directly without attributes in your answer"
                )
            else:
                raise ValueError(f"Unknown operation type: '{op_type}'")

        except Exception as e:
            msg = f"Error processing {op_type} operation due to {type(e).__name__}: {str(e)}"
            await context.io.system(
                msg,
                run_label=f"{op_type.capitalize() if op_type else 'File edit'} Failed",
                run_status=cf.RunStatus.FAILED,
            )

        if msg:
            raise OrchestratorInterruption(msg)

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        if tag.name != self.tag_name:
            return

        await self._on_file_edit_tag(tag, context)

    @property
    async def tools(self) -> list[ant.ToolLike]:
        if self.enable_tool_use:
            return [self.editor_tool]

        return []

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        with sanitize_pydantic_validation_error():
            edit_input = ant.TextEditorInput.parse_obj(tool_use.input)
        command = edit_input.command
        path = Path(edit_input.path)
        try:
            if command == ant.TextEditorCommand.CREATE:
                msg = await self._on_create_command(
                    file_path=path, edit_input=edit_input, context=context
                )
            elif command == ant.TextEditorCommand.STR_REPLACE:
                msg = await self._on_str_replace_command(
                    file_path=path, edit_input=edit_input, context=context
                )
            elif command == ant.TextEditorCommand.INSERT:
                msg = await self._on_insert_command(
                    file_path=path, edit_input=edit_input, context=context
                )
            elif command == ant.TextEditorCommand.VIEW:
                msg = await self._on_view_command(
                    file_path=path, edit_input=edit_input, context=context
                )
            elif command == ant.TextEditorCommand.UNDO_EDIT:
                msg = await self._on_undo_edit_command(
                    file_path=path, edit_input=edit_input, context=context
                )
            else:
                raise ValueError(f"Unknown command: '{command}'")

            return ant.MessageContentToolResult(tool_use_id=tool_use.id, content=msg)
        except Exception as e:
            # Store the error type for use in on_after_tool_use_result
            self._get_session_storage(context)[LAST_ERROR_TYPE_KEY] = type(e).__name__

            msg = f"Error processing `{command}` command due to {type(e).__name__}: {str(e)}"
            await context.io.system(
                msg,
                run_label=command.replace("_", " ").capitalize() + " Failed",
                run_status=cf.RunStatus.FAILED,
            )
            raise

    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        if tool_result.is_error:
            # Check what type of error occurred
            session_storage = self._get_session_storage(context)
            last_error_type = session_storage.get(LAST_ERROR_TYPE_KEY)

            # Clean up the stored error type immediately to prevent stale state
            session_storage.pop(LAST_ERROR_TYPE_KEY, None)

            # Choose appropriate message based on error type
            if last_error_type == FileAccessError.__name__:
                message_content = FILE_EDIT_TOOL_USER_ACCESS_DENIED_MESSAGE
            else:
                message_content = FILE_EDIT_TOOL_USE_USER_ERR_MESSAGE

            context.memory_manager.add_messages(
                [
                    CfMessage(
                        type=cf.MessageType.HUMAN,
                        content=message_content,
                    )
                ]
            )

    async def _on_create_command(
        self,
        file_path: Path,
        edit_input: ant.TextEditorInput,
        context: AnalectRunContext,
    ) -> str:
        if edit_input.file_text is None:
            raise ValueError('"file_text" is required for `create` command')
        return await self._on_create(
            file_path=file_path,
            content=edit_input.file_text,
            context=context,
            require_line_num=False,
        )

    async def _on_str_replace_command(
        self,
        file_path: Path,
        edit_input: ant.TextEditorInput,
        context: AnalectRunContext,
    ) -> str:
        if edit_input.old_str is None:
            raise ValueError('"old_str" is required for `str_replace` command')
        return await self._on_replace(
            file_path=file_path,
            find_text=edit_input.old_str,
            replace_text=edit_input.new_str or "",
            context=context,
            require_line_num=False,
        )

    async def _on_insert_command(
        self,
        file_path: Path,
        edit_input: ant.TextEditorInput,
        context: AnalectRunContext,
    ) -> str:
        if edit_input.insert_line is None:
            raise ValueError('"insert_line" is required for `insert` command')
        if edit_input.new_str is None:
            raise ValueError('"new_str" is required for `insert` command')

        return await self._on_insert(
            file_path=file_path,
            find_after=None,
            content=edit_input.new_str,
            context=context,
            require_line_num=False,
            insert_line=edit_input.insert_line,
        )

    async def _on_view_command(
        self,
        file_path: Path,
        edit_input: ant.TextEditorInput,
        context: AnalectRunContext,
    ) -> str:
        if file_path.is_dir():
            return await self._on_view_directory(
                directory=file_path,
                depth=2,
                show_hidden=False,
                context=context,
            )
        else:
            view_range = edit_input.view_range
            start_line = None
            end_line = None
            if view_range is not None:
                start_line = view_range[0] if len(view_range) > 0 else None
                end_line = view_range[1] if len(view_range) > 1 else None
            return await self._on_view_file_path(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                context=context,
            )

    async def _on_undo_edit_command(
        self,
        file_path: Path,
        edit_input: ant.TextEditorInput,
        context: AnalectRunContext,
    ) -> str:
        return await self._on_undo(file_path=file_path, context=context)


def _read_text_from_path(path: Path) -> str:
    if path.is_file() and path.exists():
        return path.read_text()
    return ""
