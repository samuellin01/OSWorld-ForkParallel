# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import re
import subprocess
from textwrap import dedent
from typing import Any, cast

from langchain_core.runnables import RunnableLambda
from pydantic import Field, PrivateAttr

from .....core import types as cf

from .....core.analect.analect import AnalectRunContext, get_current_context
from .....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from .....utils.string import truncate
from ....tags import Tag, TagLike
from ...tool_use import ToolUseExtension
from ..exceptions import InvalidCommandLineInput, SessionNotFoundError
from ..utils import get_allowed_and_disallowed_commands, get_command_tokens_from_bash

from .prompts import (
    INTERACT_TERMINAL_DESCRIPTION,
    LIST_TERMINALS_DESCRIPTION,
    SPAWN_TERMINAL_DESCRIPTION,
    TERMINATE_TERMINAL_DESCRIPTION,
)
from .types import (
    InteractTerminalInput,
    ListTerminalsInput,
    SpawnTerminalInput,
    TerminalOutput,
    TerminalSession,
    TerminalSessionsInfo,
    TerminateTerminalInput,
)
from .utils import (
    create_terminal,
    format_terminal_list,
    get_terminal_sessions_info,
    interact_terminal,
    terminate_terminal,
)

# Tool names
SPAWN_TERMINAL_TOOL_NAME: str = "spawn_terminal"
INTERACT_TERMINAL_TOOL_NAME: str = "interact_terminal"
TERMINATE_TERMINAL_TOOL_NAME: str = "terminate_terminal"
LIST_TERMINALS_TOOL_NAME: str = "list_terminals"

# Session storage key
TERMINAL_SESSIONS_KEY: str = "terminal_sessions"
ANSI_FILTER_PATTERN: re.Pattern[str] = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)


class PExpectTerminalExtension(ToolUseExtension):
    """Extension for interactive terminal sessions using pexpect."""

    included_in_system_prompt: bool = False
    trace_tool_execution: bool = False  # Custom trace node will be generated
    allowed_commands: dict[str, TagLike] = Field(
        default_factory=dict,
        description="Dict of allowed commands, with command name as key and description as value",
    )
    disallowed_commands: dict[str, TagLike] = Field(
        default_factory=dict,
        description="Dict of explicitly disallowed commands, with command name as key and reason as value",
    )
    max_output_lines: int = Field(
        default=100,
        description="Maximum number of lines of the output to include in the response string",
    )
    cwd: str | None = Field(
        None, description="Current working directory for terminal sessions"
    )
    env: dict[str, str] | None = Field(
        None, description="Environment variables for terminal sessions"
    )
    # TODO: explore color rendering using https://fburl.com/code/ezyqf8t1
    filter_ansi: bool = Field(
        default=True,
        description="Whether to filter out ANSI escape sequences from the output",
    )
    _tokenized_allowed_commands: list[list[str]] = PrivateAttr([])
    _tokenized_disallowed_commands: list[list[str]] = PrivateAttr([])

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._tokenized_allowed_commands = [
            tokens
            for command in self.allowed_commands
            for tokens in get_command_tokens_from_bash(command)
        ]
        self._tokenized_disallowed_commands = [
            tokens
            for command in self.disallowed_commands
            for tokens in get_command_tokens_from_bash(command)
        ]

    @property
    async def tools(self) -> list[ant.ToolLike]:
        """Return the tools provided by this extension."""
        if not self.enable_tool_use:
            return []

        spawn_desc = SPAWN_TERMINAL_DESCRIPTION
        if self.allowed_commands:
            spawn_desc += (
                "\n\n"
                + Tag(
                    name="allowed_commands",
                    contents=[
                        Tag(
                            name="command",
                            attributes={"name": name},
                            contents=desc,
                        )
                        for name, desc in self.allowed_commands.items()
                    ],
                ).prettify()
            )

        return [
            # 1. Spawn Terminal Tool
            ant.Tool(
                name=SPAWN_TERMINAL_TOOL_NAME,
                description=spawn_desc,
                input_schema=SpawnTerminalInput.model_json_schema(),
            ),
            # 2. Interact Terminal Tool
            ant.Tool(
                name=INTERACT_TERMINAL_TOOL_NAME,
                description=INTERACT_TERMINAL_DESCRIPTION,
                input_schema=InteractTerminalInput.model_json_schema(),
            ),
            # 3. Terminate Terminal Tool
            ant.Tool(
                name=TERMINATE_TERMINAL_TOOL_NAME,
                description=TERMINATE_TERMINAL_DESCRIPTION,
                input_schema=TerminateTerminalInput.model_json_schema(),
            ),
            # 4. List Terminals Tool
            ant.Tool(
                name=LIST_TERMINALS_TOOL_NAME,
                description=LIST_TERMINALS_DESCRIPTION,
                input_schema=ListTerminalsInput.model_json_schema(),
            ),
        ]

    @property
    def _terminal_sessions(self) -> dict[int, TerminalSession]:
        """Get the terminal sessions from session storage."""
        context = get_current_context()
        namespace = self.__class__.__name__
        return cast(
            dict[int, TerminalSession],
            context.session_storage[namespace].setdefault(TERMINAL_SESSIONS_KEY, {}),
        )

    def _get_session(self, session_id: int) -> TerminalSession:
        """Get a terminal session by its ID."""
        session = self._terminal_sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(
                f"No terminal session found with process ID: {session_id}"
            )
        return session

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed based on the allowed_commands dict."""
        if not self.allowed_commands:
            return True
        return any(cmd in command for cmd in self.allowed_commands.keys())

    def _get_fresh_environment(self, inp: SpawnTerminalInput) -> None:
        """Get fresh environment variables from subprocess and update inp.env.

        This method gets the current environment state by running 'env -0'
        subprocess command, which provides more up-to-date environment
        variables than os.environ. Custom inp.env values take precedence.

        Args:
            inp: SpawnTerminalInput to update with fresh environment
        """
        # Use -0 flag to separate entries with null bytes, making multiline values unambiguous
        env_result = subprocess.run(["env", "-0"], capture_output=True, text=True)
        fresh_env = {}

        # Parse environment output using null byte separator
        for entry in env_result.stdout.strip("\0").split("\0"):
            if entry and "=" in entry:
                key, value = entry.split("=", 1)
                fresh_env[key] = value

        # If custom env exists, merge it with fresh env (custom takes precedence)
        if inp.env is not None:
            fresh_env.update(inp.env)

        inp.env = fresh_env

    async def spawn_terminal(
        self, inp: SpawnTerminalInput, context: AnalectRunContext
    ) -> TerminalOutput:
        """Handle spawning a new terminal session."""

        runnable = RunnableLambda(self._spawn_terminal, name=inp.name + " (spawn)")
        return await context.invoke(runnable, inp, run_type="tool")

    async def on_before_spawn_terminal(
        self,
        inp: SpawnTerminalInput,
        context: AnalectRunContext,
    ) -> SpawnTerminalInput:
        """
        This method is called before spawning a new terminal session. It can be used to perform any necessary pre-processing or validation before the session creation.
        """
        await context.io.system(
            dedent(
                """\
                Starting terminal session in {cwd}:
                ```console
                {command}
                ```
                """
            ).format(command=inp.command, cwd=f"`{inp.cwd}`" or "default cwd"),
            run_label=f"Spawning terminal: {inp.name}",
        )
        return inp

    async def on_spawn_terminal_success(
        self,
        inp: SpawnTerminalInput,
        out: TerminalOutput,
        context: AnalectRunContext,
    ) -> TerminalOutput:
        """
        This method is called after a successful terminal session creation. It can be used to perform any necessary post-processing or logging after the session creation.
        """

        # Log the result
        if out.is_alive:
            status = cf.RunStatus.COMPLETED
            label_suffix = "Started"
        else:
            # Check exit_status to determine if the terminal completed successfully or failed
            if out.exit_status == 0:
                status = cf.RunStatus.COMPLETED
                label_suffix = "Completed"
            else:
                status = cf.RunStatus.FAILED
                label_suffix = f"Failed (exit code: {out.exit_status})"

        await context.io.system(
            out.to_markdown(),
            run_label=f"Terminal {inp.name} {label_suffix}",
            run_status=status,
        )
        return out

    async def on_spawn_terminal_failed(
        self,
        inp: SpawnTerminalInput,
        exception: Exception,
        context: AnalectRunContext,
    ) -> None:
        """
        This method is called after a terminal session creation fails. It can be used to
        perform any necessary error handling or logging after the session creation failure.
        """
        await context.io.system(
            f"Failed to spawn terminal: {inp.command}\n\nError: {str(exception)}",
            run_label=f"Terminal {inp.name} failed",
            run_status=cf.RunStatus.FAILED,
        )

    async def _validate_command(self, command: str) -> None:
        """
        Validate the command before running it.
        """
        result = get_allowed_and_disallowed_commands(
            command,
            self._tokenized_allowed_commands,
            self._tokenized_disallowed_commands,
        )

        # First check for explicitly disallowed commands
        if result.explicitly_disallowed:
            disallowed_cmd = list(result.explicitly_disallowed)[
                0
            ]  # Get the first disallowed command
            reason = self.disallowed_commands.get(
                disallowed_cmd, "This command is explicitly disallowed"
            )
            raise InvalidCommandLineInput(
                f"`{command}` uses command that is explicitly disallowed: `{disallowed_cmd}`. Reason: {reason}"
            )

        # Then check for commands that aren't in the allowed list
        if result.disallowed:
            raise InvalidCommandLineInput(
                f"`{command}` uses commands that aren't allowed: `{'`,`'.join(result.disallowed)}`. Please use only allowed commands: {','.join(self.allowed_commands.keys())}"
            )

    async def _spawn_terminal(self, inp: SpawnTerminalInput) -> TerminalOutput:
        # Validate command against allowed commands if specified
        context = get_current_context()
        await context.io.system(
            f"Validating command `{inp.command}`", run_label="Validating command"
        )
        try:
            await self._validate_command(command=inp.command)
        except InvalidCommandLineInput as e:
            await context.io.system(
                str(e), run_label="Validation failed", run_status=cf.RunStatus.FAILED
            )
            raise e

        if inp.cwd is None and self.cwd is not None:
            inp.cwd = self.cwd

        self._get_fresh_environment(inp)

        if inp.env is None and self.env is not None:
            inp.env = self.env

        try:
            inp = await self.on_before_spawn_terminal(inp, context)
            output = await create_terminal(inp, self._terminal_sessions)

            if self.filter_ansi:
                output.output = ANSI_FILTER_PATTERN.sub("", output.output)

            output.output = truncate(output.output, self.max_output_lines)
            return await self.on_spawn_terminal_success(inp, output, context)
        except Exception as e:
            await self.on_spawn_terminal_failed(inp, e, context)
            raise

    async def interact_terminal(
        self, inp: InteractTerminalInput, context: AnalectRunContext
    ) -> TerminalOutput:
        """Handle interacting with an existing terminal session."""
        session = self._terminal_sessions.get(inp.session_id)
        runnable = RunnableLambda(
            self._interact_terminal,
            name=str(inp.session_id)
            if session is None
            else session.input.name + " (interact)",
        )
        return await context.invoke(runnable, inp, run_type="tool")

    def _get_interact_action(self, inp: InteractTerminalInput) -> str:
        action = "Checking output"
        if inp.send_control:
            action = f"Sending Ctrl+{inp.send_control.upper()}"
        elif inp.send_eof:
            action = "Sending EOF"
        elif inp.send is not None:
            action = "Sending input"
        elif inp.send_line is not None:
            action = "Sending input line"
        return action

    async def on_before_interact_terminal(
        self,
        inp: InteractTerminalInput,
        context: AnalectRunContext,
    ) -> InteractTerminalInput:
        """
        This method is called before interacting with a terminal session. It can be used to perform
        any necessary pre-processing or validation before the terminal interaction.
        """
        session = self._get_session(inp.session_id)
        action = self._get_interact_action(inp)

        await context.io.system(
            f"{action} to terminal `{session.input.name}` (ID: {inp.session_id})\n"
            + (
                dedent(
                    """\
                ```console
                {input_text}
                ```
                """
                ).format(input_text=inp.send or inp.send_line or "")
            ),
            run_label=f"{action} to {session.input.name}",
        )
        return inp

    async def on_interact_terminal_success(
        self,
        inp: InteractTerminalInput,
        out: TerminalOutput,
        session: TerminalSession,
        context: AnalectRunContext,
    ) -> TerminalOutput:
        """
        This method is called after a successful terminal interaction. It can be used to perform
        any necessary post-processing or logging after the terminal interaction.
        """
        session = self._get_session(inp.session_id)
        action = self._get_interact_action(inp)

        if not out.is_alive:
            # Remove from sessions if terminated
            self._terminal_sessions.pop(inp.session_id, None)

            # Check exit_status to determine run status
            status = (
                cf.RunStatus.COMPLETED if out.exit_status == 0 else cf.RunStatus.FAILED
            )
            status_label = (
                "completed"
                if out.exit_status == 0
                else f"failed (exit code: {out.exit_status})"
            )
        else:
            status = cf.RunStatus.COMPLETED
            status_label = "completed"

        await context.io.system(
            out.to_markdown(),
            run_label=f"{action} to {session.input.name} {status_label}",
            run_status=status,
        )
        return out

    async def on_interact_terminal_failed(
        self,
        inp: InteractTerminalInput,
        exception: Exception,
        context: AnalectRunContext,
    ) -> None:
        """
        This method is called after a terminal interaction fails. It can be used to
        perform any necessary error handling or logging after the interaction failure.
        """
        session = self._terminal_sessions.get(inp.session_id)
        sess_name = str(inp.session_id) if session is None else session.input.name
        action = self._get_interact_action(inp)

        await context.io.system(
            f"Failed to interact with terminal session {inp.session_id}\n\nError: {str(exception)}",
            run_label=f"{action} to {sess_name} failed",
            run_status=cf.RunStatus.FAILED,
        )

    async def _interact_terminal(self, inp: InteractTerminalInput) -> TerminalOutput:
        # Find the terminal session
        session = self._get_session(inp.session_id)
        context = get_current_context()

        try:
            inp = await self.on_before_interact_terminal(inp, context)
            # Interact with the terminal
            output = await interact_terminal(inp, session)

            if self.filter_ansi:
                output.output = ANSI_FILTER_PATTERN.sub("", output.output)

            output.output = truncate(output.output, self.max_output_lines)
            return await self.on_interact_terminal_success(
                inp, output, session, context
            )
        except Exception as e:
            await self.on_interact_terminal_failed(inp, e, context)
            raise

    async def on_terminate_terminal(
        self, inp: TerminateTerminalInput, context: AnalectRunContext
    ) -> TerminalOutput:
        """Handle terminating a terminal session."""
        session = self._terminal_sessions.get(inp.session_id)
        runnable = RunnableLambda(
            self._terminate_terminal,
            name=str(inp.session_id)
            if session is None
            else session.input.name + " (terminate)",
        )
        return await context.invoke(runnable, inp, run_type="tool")

    async def on_before_terminate_terminal(
        self,
        inp: TerminateTerminalInput,
        session: TerminalSession,
        context: AnalectRunContext,
    ) -> TerminateTerminalInput:
        """
        This method is called before terminating a terminal session. It can be used to perform
        any necessary pre-processing or validation before the session termination.
        """
        await context.io.system(
            f"Terminating terminal session: {session.input.name} (ID: {inp.session_id}){' (forced)' if inp.force else ''}",
            run_label=f"Terminating terminal: {session.input.name}",
        )
        return inp

    async def on_terminate_terminal_success(
        self,
        inp: TerminateTerminalInput,
        out: TerminalOutput,
        session: TerminalSession,
        context: AnalectRunContext,
    ) -> TerminalOutput:
        """
        This method is called after a successful terminal termination. It can be used to perform
        any necessary post-processing or logging after the session termination.
        """
        # Check exit_status to determine run status
        status = cf.RunStatus.COMPLETED if out.exit_status == 0 else cf.RunStatus.FAILED
        status_label = (
            "terminated"
            if out.exit_status == 0
            else f"terminated with error (exit code: {out.exit_status})"
        )

        await context.io.system(
            out.to_markdown(),
            run_label=f"Terminal {session.input.name} {status_label}",
            run_status=status,
        )

        # Remove from sessions
        self._terminal_sessions.pop(inp.session_id, None)

        return out

    async def on_terminate_terminal_failed(
        self,
        inp: TerminateTerminalInput,
        exception: Exception,
        context: AnalectRunContext,
    ) -> None:
        """
        This method is called after a terminal termination fails. It can be used to
        perform any necessary error handling or logging after the termination failure.
        """
        await context.io.system(
            f"Failed to terminate terminal session {inp.session_id}\n\nError: {str(exception)}",
            run_label="Terminal termination failed",
            run_status=cf.RunStatus.FAILED,
        )

    async def _terminate_terminal(self, inp: TerminateTerminalInput) -> TerminalOutput:
        # Find the terminal session
        session = self._get_session(inp.session_id)
        context = get_current_context()

        try:
            inp = await self.on_before_terminate_terminal(inp, session, context)
            # Terminate the session
            output = await terminate_terminal(session, inp.force)

            if self.filter_ansi:
                output.output = ANSI_FILTER_PATTERN.sub("", output.output)

            output.output = truncate(output.output, self.max_output_lines)
            return await self.on_terminate_terminal_success(
                inp, output, session, context
            )
        except Exception as e:
            await self.on_terminate_terminal_failed(inp, e, context)
            raise

    async def list_terminals(
        self, inp: ListTerminalsInput, context: AnalectRunContext
    ) -> TerminalSessionsInfo:
        """Handle listing terminal sessions."""
        runnable = RunnableLambda(
            self._list_terminals,
            name="list_terminals",
        )
        return await context.invoke(runnable, inp, run_type="tool")

    async def on_before_list_terminals(
        self,
        inp: ListTerminalsInput,
        context: AnalectRunContext,
    ) -> ListTerminalsInput:
        """
        This method is called before listing terminal sessions. It can be used to perform
        any necessary pre-processing or validation before listing the sessions.
        """
        await context.io.system(
            "Listing active terminal sessions",
            run_label="Listing terminals",
        )
        return inp

    async def on_list_terminals_success(
        self,
        inp: ListTerminalsInput,
        out: TerminalSessionsInfo,
        terminal_list: str,
        context: AnalectRunContext,
    ) -> TerminalSessionsInfo:
        """
        This method is called after successfully listing terminal sessions. It can be used to perform
        any necessary post-processing or logging after listing the sessions.
        """
        await context.io.system(
            terminal_list,
            run_label="Terminal sessions info retrieved",
            run_status=cf.RunStatus.COMPLETED,
        )
        return out

    async def on_list_terminals_failed(
        self,
        inp: ListTerminalsInput,
        exception: Exception,
        context: AnalectRunContext,
    ) -> None:
        """
        This method is called after listing terminal sessions fails. It can be used to
        perform any necessary error handling or logging after the listing failure.
        """
        await context.io.system(
            f"Failed to list terminal sessions\n\nError: {str(exception)}",
            run_label="Terminal listing failed",
            run_status=cf.RunStatus.FAILED,
        )

    async def _list_terminals(self, inp: ListTerminalsInput) -> TerminalSessionsInfo:
        context = get_current_context()

        try:
            inp = await self.on_before_list_terminals(inp, context)

            # Clean up any terminated sessions
            for session_id in list(self._terminal_sessions.keys()):
                session = self._terminal_sessions[session_id]
                if not session.is_alive():
                    self._terminal_sessions.pop(session_id, None)

            # Generate the markdown list for logging
            terminal_list = format_terminal_list(self._terminal_sessions)

            # Return detailed information about sessions in JSON format
            out = get_terminal_sessions_info(self._terminal_sessions)

            return await self.on_list_terminals_success(
                inp, out, terminal_list, context
            )
        except Exception as e:
            await self.on_list_terminals_failed(inp, e, context)
            raise

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Handle tool usage."""
        is_error = False
        if tool_use.name == SPAWN_TERMINAL_TOOL_NAME:
            # Handle spawn terminal
            inp = SpawnTerminalInput.model_validate(tool_use.input)
            out = await self.spawn_terminal(inp, context)
            is_error = out.exit_status is not None and out.exit_status != 0

        elif tool_use.name == INTERACT_TERMINAL_TOOL_NAME:
            # Handle interact terminal
            inp = InteractTerminalInput.model_validate(tool_use.input)
            out = await self.interact_terminal(inp, context)

        elif tool_use.name == TERMINATE_TERMINAL_TOOL_NAME:
            # Handle terminate terminal
            inp = TerminateTerminalInput.model_validate(tool_use.input)
            out = await self.on_terminate_terminal(inp, context)

        elif tool_use.name == LIST_TERMINALS_TOOL_NAME:
            # Handle list terminals
            inp = ListTerminalsInput.model_validate(tool_use.input)
            out = await self.list_terminals(inp, context)

        else:
            raise ValueError(f"Unknown tool: {tool_use.name}")

        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content=out.model_dump_json(),
            is_error=is_error,
        )
