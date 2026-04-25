# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from pathlib import Path
from textwrap import dedent
from typing import Any, cast, Dict, List, Optional

from pydantic import BaseModel, Field

from .....core import types as cf
from .....core.analect import AnalectRunContext

from .....core.chat_models.bedrock.api.invoke_model import anthropic as ant
# Custom IO types not available in this version

from .....utils.artifact import set_artifact
from ....tags import TagLike
from ...file.utils import replace_in_file, view_directory, view_file_content
from ...function.utils import Function, get_runnable
from ..reminder import MemoryReminder

from . import utils
from .exceptions import MemoryNodeNotFoundError
from .prompts import (
    DELETE_MEMORY_DESCRIPTION,
    EDIT_MEMORY_DESCRIPTION,
    HIERARCHICAL_MEMORY_DESCRIPTION,
    HIERARCHICAL_MEMORY_REMINDER_MESSAGE,
    IMPORT_MEMORY_DESCRIPTION,
    READ_MEMORY_DESCRIPTION,
    SEARCH_MEMORY_DESCRIPTION,
    WRITE_MEMORY_DESCRIPTION,
)
from .types import Memory, MemoryNode

NUM_LLM_CALLS_KEY = "num_llm_calls"


class SearchMemoryInput(BaseModel):
    path_pattern: Optional[str] = Field(
        None,
        description="Pattern to match memory node names in the hierarchy (supports glob patterns like 'project/*' or 'notes/*.md')",
    )
    content_pattern: Optional[str] = Field(
        None,
        description="Text pattern to search within memory content (supports regex)",
    )
    tags: Optional[List[str]] = Field(
        None, description="Filter by tags (all specified tags must be present)"
    )
    max_results: int = Field(
        default=20, description="Maximum number of results to return"
    )


class ReadMemoryInput(BaseModel):
    path: str = Field(
        ...,
        description="Path to memory node to read. For files, must end with '.md' (e.g., 'project/notes.md'). For directories, use the directory path without '.md' (e.g., 'project/')",
    )
    start_line: Optional[int] = Field(None, description="Start line for partial read")
    end_line: Optional[int] = Field(None, description="End line for partial read")


class EditMemoryInput(BaseModel):
    path: str = Field(
        ...,
        description="Path to memory node to edit. Must end with '.md' (e.g., 'project/notes.md')",
    )
    old_str: str = Field(..., description="Exact text to find and replace")
    new_str: str = Field(..., description="Replacement text to insert")


class WriteMemoryInput(BaseModel):
    path: str = Field(
        ...,
        description="Path for the memory node. Must end with '.md' (e.g., 'project/notes.md')",
    )
    content: str = Field(..., description="Markdown content to store")
    tags: List[str] = Field(
        default_factory=list, description="Tags to associate with this memory node"
    )


class DeleteMemoryInput(BaseModel):
    paths: List[str] = Field(
        ...,
        description="List of memory node paths to delete (will delete children too). File paths must end with '.md' (e.g., 'project/notes.md'), directory paths should not end with '.md' (e.g., 'project/')",
    )


class ImportMemoryInput(BaseModel):
    session_uuids: List[str] = Field(
        ...,
        description="List of UUIDs of the source sessions to import memory from",
    )


class MemoryOperationResult(BaseModel):
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Result message or error description")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional result data")


class HierarchicalMemoryExtension(MemoryReminder):
    """Hierarchical memory extension with file-system based organization."""

    name: str = "hierarchical_memory"
    included_in_system_prompt: bool = True
    trace_tool_execution: bool = False
    namespace: str = Field(
        default="hierarchical_memory", description="Namespace for the memory directory"
    )
    directory: Path = Field(
        default=Path("/tmp/confucius_memory"),
        description="Base directory to store memory files",
    )
    artifact_identifier: str = Field(
        default="hierarchical_memory", description="Identifier for the memory artifact"
    )
    artifact_display_name: str = Field(
        default="Memory", description="Display name for the memory artifact"
    )
    # Configuration fields for local storage
    reminder_message: str = Field(
        default=HIERARCHICAL_MEMORY_REMINDER_MESSAGE,
        description="Message to remind the user to write or edit hierarchical memory",
    )

    async def description(self) -> TagLike:
        return HIERARCHICAL_MEMORY_DESCRIPTION

    @property
    async def tools(self) -> List[ant.ToolLike]:
        if self.enable_tool_use:
            tools = await super().tools
            return tools + [
                ant.Tool(
                    name="search_memory",
                    description=SEARCH_MEMORY_DESCRIPTION,
                    input_schema=SearchMemoryInput.model_json_schema(),
                ),
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
                ant.Tool(
                    name="delete_memory",
                    description=DELETE_MEMORY_DESCRIPTION,
                    input_schema=DeleteMemoryInput.model_json_schema(),
                ),
                ant.Tool(
                    name="import_memory",
                    description=IMPORT_MEMORY_DESCRIPTION,
                    input_schema=ImportMemoryInput.model_json_schema(),
                ),
            ]
        return []

    def _get_memory_base_dir(self, context: AnalectRunContext) -> Path:
        """Get the base directory for this session's memory."""
        session_id = context.session or "default"
        session_dir = self.directory / f"{self.namespace}_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    # Memory synchronization methods not implemented in this version

    async def _load_memory_from_filesystem(self, context: AnalectRunContext) -> Memory:
        """Load memory structure by scanning the filesystem."""

        base_dir = self._get_memory_base_dir(context)
        if not base_dir.exists():
            return Memory()

        def build_node_from_path(path: Path, node_name: str) -> Optional[MemoryNode]:
            if path.is_file() and path.suffix == ".md":
                # Read file content and parse frontmatter
                content = path.read_text()
                tags, clean_content = utils.parse_frontmatter(content)

                # Keep full node name with .md extension for display
                display_name = node_name
                return MemoryNode(
                    path=path,  # Full filesystem path
                    name=display_name,  # Full display name with .md extension
                    content=clean_content,
                    tags=tags,
                    children=[],
                )
            elif path.is_dir():
                # Create directory node with children
                children = []
                for child_path in sorted(path.iterdir()):
                    child_node = build_node_from_path(child_path, child_path.name)
                    if child_node:
                        children.append(child_node)

                if children:  # Only create directory nodes if they have children
                    return MemoryNode(
                        path=path,  # Full filesystem path
                        name=node_name,  # Simple directory name
                        content="",
                        tags=[],
                        children=children,
                    )

            return None

        # Build nodes from filesystem
        nodes = []
        for path in sorted(base_dir.iterdir()):
            node = build_node_from_path(path, path.name)
            if node:
                # Apply folder collapsing
                merged_node = utils.merge_single_child_memory_dirs(node)
                nodes.append(merged_node)

        return Memory(nodes=nodes)

    def _get_content_file_path(
        self, node_path: str, context: AnalectRunContext
    ) -> Path:
        """Get the filesystem path for a memory node's content."""
        base_dir = self._get_memory_base_dir(context)
        return base_dir / node_path

    async def _display_memory(self, memory: Memory, context: AnalectRunContext) -> None:
        """Display the current memory structure as an artifact."""
        self.reset_reminder()
        try:
            # Create a simple artifact with memory structure
            attachment = await set_artifact(
                name=self.artifact_identifier,
                value=memory.model_dump_json(),  # Simple JSON representation
                display_name=self.artifact_display_name,
            )
            await context.io.ai("Memory updated", attachments=[attachment])
        except Exception as e:
            await context.io.system(
                f"Failed to display memory due to {type(e).__name__}: {str(e)}",
                run_label="Display Memory",
                run_status=cf.RunStatus.FAILED,
            )

    async def _run_func(
        self,
        func: Function,
        context: AnalectRunContext,
        /,
        **kwargs: Any,
    ) -> object:
        """
        Run a function with the given arguments and return the result.

        Args:
            func (Function): The function to run.
            context (AnalectRunContext): The context of the run.

        Returns:
            object: The result of the function.
        """

        runnable = get_runnable(func)
        return await context.invoke(runnable, kwargs, run_type="tool")

    async def _search_memory(
        self, inp: SearchMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Search for memory nodes based on various criteria."""
        await context.io.system(
            dedent("""\
            Searching memory with the following criteria:
            
            ```json
            {data}
            ```
            """).format(data=inp.model_dump_json(indent=2)),
            run_label="Searching Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        memory: Memory = await self._load_memory_from_filesystem(context)
        base_dir: Path = self._get_memory_base_dir(context)

        def search_memory(
            path_pattern: Optional[str],
            content_pattern: Optional[str],
            tags: Optional[List[str]],
            max_results: int,
        ) -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []

            utils.collect_matching_nodes(
                memory.nodes,
                path_pattern,
                content_pattern,
                tags,
                max_results,
                results,
                base_dir,
            )
            return results

        results = cast(
            list[dict[str, object]],
            await self._run_func(
                search_memory,
                context,
                path_pattern=inp.path_pattern,
                content_pattern=inp.content_pattern,
                tags=inp.tags,
                max_results=inp.max_results,
            ),
        )

        await context.io.system(
            f"Found {len(results)} matching memory nodes",
            run_label="Searching Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        return MemoryOperationResult(
            success=True,
            message=f"Found {len(results)} matching memory nodes",
            data={"results": results},
        )

    async def _read_memory(
        self, inp: ReadMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Read content from a specific memory node."""
        await context.io.system(
            f"Reading memory node '{inp.path}'...",
            run_label="Reading Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        def read_memory(
            path: str,
            start_line: Optional[int],
            end_line: Optional[int],
        ) -> str:
            # Get the content file path
            content_file = self._get_content_file_path(path, context)

            if not content_file.exists():
                raise MemoryNodeNotFoundError(f"Memory node '{path}' not found")

            if content_file.is_dir():
                return view_directory(
                    content_file,
                    depth=1,
                    show_hidden=False,
                )

            # Use view_file_content for line range support
            return view_file_content(
                content_file.read_text(),
                start_line=start_line,
                end_line=end_line,
                max_view_lines=None,
                include_line_numbers=True,
            )

        content = cast(
            str,
            await self._run_func(
                read_memory,
                context,
                path=inp.path,
                start_line=inp.start_line,
                end_line=inp.end_line,
            ),
        )

        await context.io.system(
            f"Successfully read memory node '{inp.path}'",
            run_label="Reading Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        return MemoryOperationResult(
            success=True,
            message=f"Successfully read memory node '{inp.path}'",
            data={"path": inp.path, "content": content},
        )

    async def _write_memory(
        self, inp: WriteMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Write content to a memory node."""
        await context.io.system(
            f"Writing to memory node '{inp.path}'...",
            run_label="Writing Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        def write_memory(
            path: str,
            content: str,
            tags: List[str],
        ) -> dict[str, object]:
            # Get the content file path
            content_file = self._get_content_file_path(path, context)

            # Check if node exists
            is_update = content_file.exists()

            # Create parent directories if needed
            content_file.parent.mkdir(parents=True, exist_ok=True)

            # Create content with frontmatter
            content_with_frontmatter = utils.create_content_with_frontmatter(
                content, tags
            )

            # Write to file
            content_file.write_text(content_with_frontmatter)

            action = "Updated" if is_update else "Created"
            return {"action": action.lower(), "path": path}

        result = cast(
            dict[str, object],
            await self._run_func(
                write_memory,
                context,
                path=inp.path,
                content=inp.content,
                tags=inp.tags,
            ),
        )

        action = "Updated" if result["action"] == "updated" else "Created"
        message = (
            f"{action} memory node '{inp.path}' with {len(inp.content)} characters"
        )
        if inp.tags:
            message += f" and tags: {', '.join(inp.tags)}"

        # Display updated memory
        memory = await self._load_memory_from_filesystem(context)

        await context.io.system(
            message,
            run_label="Writing Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        await self._display_memory(memory, context)

        return MemoryOperationResult(
            success=True,
            message=message,
        )

    async def _delete_memory(
        self, inp: DeleteMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Delete memory nodes and their children."""
        await context.io.system(
            f"Deleting {len(inp.paths)} memory nodes...",
            run_label="Deleting Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        def delete_memory(
            paths: List[str],
        ) -> dict[str, object]:
            deleted = []
            not_found = []

            # Process each path
            for path in paths:
                # Get the content file path
                content_file = self._get_content_file_path(path, context)

                if not content_file.exists():
                    not_found.append(path)
                    continue

                # Delete the file
                content_file.unlink()

                # Clean up empty parent directories
                base_dir = self._get_memory_base_dir(context)
                utils.cleanup_empty_parent_directories(content_file, base_dir)

                deleted.append(path)

            return {"deleted": deleted, "not_found": not_found}

        result = cast(
            dict[str, object],
            await self._run_func(
                delete_memory,
                context,
                paths=inp.paths,
            ),
        )

        deleted = cast(List[str], result["deleted"])
        not_found = cast(List[str], result["not_found"])

        # Prepare result message
        message_parts = []
        if deleted:
            message_parts.append(
                f"Deleted {len(deleted)} memory nodes: {', '.join(deleted)}"
            )
        if not_found:
            message_parts.append(
                f"Could not find {len(not_found)} nodes: {', '.join(not_found)}"
            )

        message = "; ".join(message_parts)
        status = cf.RunStatus.COMPLETED if deleted else cf.RunStatus.FAILED

        await context.io.system(
            message,
            run_label="Deleting Memory",
            run_status=status,
        )

        # Update memory display if any nodes were deleted
        if deleted:
            memory = await self._load_memory_from_filesystem(context)
            await self._display_memory(memory, context)

        return MemoryOperationResult(
            success=len(deleted) > 0,
            message=message,
            data=result,
        )

    def _build_memory_import_summary(self, memory: Memory) -> Dict[str, Any]:
        """Build a comprehensive summary of imported memory for tool result."""
        memory_files: List[Dict[str, Any]] = []
        total_files: int = 0
        total_size: int = 0

        def collect_node_info(node: MemoryNode, parent_path: str = "") -> None:
            nonlocal total_files, total_size
            if node.content:  # This is a file node with content
                # Build the logical path (without filesystem details)
                logical_path = f"{parent_path}/{node.name}".lstrip("/")

                # Count content size
                content_size = len(node.content)
                total_size += content_size
                total_files += 1

                memory_files.append(
                    {
                        "path": logical_path,
                        "tags": node.tags,
                        "content_size": content_size,
                        "content": node.content,  # Full content for immediate access
                    }
                )

            # Process children
            for child in node.children:
                child_parent_path = (
                    f"{parent_path}/{node.name}".lstrip("/")
                    if node.name
                    else parent_path
                )
                collect_node_info(child, child_parent_path)

        # Collect information from all nodes
        for node in memory.nodes:
            collect_node_info(node)

        # Sort files by path for consistent ordering
        memory_files.sort(key=lambda x: x["path"])

        return {
            "total_files": total_files,
            "total_size": total_size,
            "memory_files": memory_files,
            "summary": f"Imported {total_files} memory files ({total_size} characters total)",
        }

    async def _import_memory(
        self, inp: ImportMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Import memory from multiple sessions."""
        session_count = len(inp.session_uuids)
        await context.io.system(
            f"Importing memory from {session_count} session(s): {', '.join(inp.session_uuids[:3])}{'...' if session_count > 3 else ''}",
            run_label="Importing Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        async def import_memory(session_uuids: List[str]) -> None:
            # Copy memory from local session directories
            import shutil

            target_dir = self._get_memory_base_dir(context)

            for i, session_uuid in enumerate(session_uuids):
                await context.io.system(
                    f"Copying memory from local session '{session_uuid}' ({i+1}/{len(session_uuids)})",
                    run_label="Importing Memory",
                    run_status=cf.RunStatus.IN_PROGRESS,
                )

                # Source directory for the session to import from
                source_session_dir = self.directory / f"{self.namespace}_{session_uuid}"

                if source_session_dir.exists():
                    # Copy entire source directory contents into target directory
                    # NOTE: When file name conflicts occur, the latest file will be kept
                    shutil.copytree(source_session_dir, target_dir, dirs_exist_ok=True)
                else:
                    await context.io.system(
                        f"Warning: No local memory found for session '{session_uuid}' (expected at {source_session_dir})",
                        run_label="Importing Memory",
                        run_status=cf.RunStatus.IN_PROGRESS,
                    )

        try:
            await self._run_func(
                import_memory,
                context,
                session_uuids=inp.session_uuids,
            )

            # Prepare detailed import summary data

            # Reload and display the updated memory
            memory = await self._load_memory_from_filesystem(context)
            memory_summary = self._build_memory_import_summary(memory)

            success_msg = (
                f"Successfully imported memory from {session_count} session(s). "
                f"Found {memory_summary['total_files']} memory files."
            )
            await context.io.system(
                success_msg,
                run_label="Importing Memory",
                run_status=cf.RunStatus.COMPLETED,
            )

            await self._display_memory(memory, context)

            return MemoryOperationResult(
                success=True,
                message=success_msg,
                data=memory_summary,
            )

        except Exception as e:
            error_msg = (
                f"Failed to import memory from {session_count} session(s): {str(e)}"
            )
            await context.io.system(
                error_msg,
                run_label="Importing Memory",
                run_status=cf.RunStatus.FAILED,
            )

            return MemoryOperationResult(
                success=False,
                message=error_msg,
            )

    async def _edit_memory(
        self, inp: EditMemoryInput, context: AnalectRunContext
    ) -> MemoryOperationResult:
        """Edit memory using file utilities."""
        await context.io.system(
            f"Editing memory node '{inp.path}'...",
            run_label="Editing Memory",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

        def edit_memory(
            path: str,
            old_str: str,
            new_str: str,
        ) -> None:
            content_file = self._get_content_file_path(path, context)

            if not content_file.exists():
                raise MemoryNodeNotFoundError(
                    f"Memory node '{path}' not found, cannot edit."
                )

            replace_in_file(
                path=content_file,
                find_text=old_str,
                replace_text=new_str,
                require_line_num=False,
            )

        await self._run_func(
            edit_memory,
            context,
            path=inp.path,
            old_str=inp.old_str,
            new_str=inp.new_str,
        )

        success_msg = (
            f"Memory edited successfully: replaced '{inp.old_str}' with '{inp.new_str}'"
        )
        await context.io.system(
            success_msg,
            run_label="Editing Memory",
            run_status=cf.RunStatus.COMPLETED,
        )

        # Display updated memory
        memory = await self._load_memory_from_filesystem(context)
        await self._display_memory(memory, context)

        return MemoryOperationResult(
            success=True,
            message=success_msg,
        )

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Handle tool usage."""
        try:
            if tool_use.name == "search_memory":
                inp = SearchMemoryInput.model_validate(tool_use.input)
                result = await self._search_memory(inp, context)

            elif tool_use.name == "read_memory":
                inp = ReadMemoryInput.model_validate(tool_use.input)
                result = await self._read_memory(inp, context)

            elif tool_use.name == "write_memory":
                inp = WriteMemoryInput.model_validate(tool_use.input)
                result = await self._write_memory(inp, context)

            elif tool_use.name == "edit_memory":
                inp = EditMemoryInput.model_validate(tool_use.input)
                result = await self._edit_memory(inp, context)

            elif tool_use.name == "delete_memory":
                inp = DeleteMemoryInput.model_validate(tool_use.input)
                result = await self._delete_memory(inp, context)

            elif tool_use.name == "import_memory":
                inp = ImportMemoryInput.model_validate(tool_use.input)
                result = await self._import_memory(inp, context)

            else:
                # Delegate to parent class for unknown tools (e.g., snooze_reminder)
                return await super().on_tool_use(tool_use, context)

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result.model_dump_json(),
                is_error=not result.success,
            )

        except Exception as e:
            msg = f"{tool_use.name} tool exection failed: due to {type(e).__name__}: {str(e)}"
            await context.io.system(
                msg,
                run_label=tool_use.name.replace("_", " ").capitalize() + " Failed",
                run_status=cf.RunStatus.FAILED,
            )
            raise
