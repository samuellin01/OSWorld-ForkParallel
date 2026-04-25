# pyre-strict

import subprocess
from textwrap import dedent
from typing import Any, Dict

import bs4

from langchain_core.runnables import RunnableLambda
from pydantic import Field, PrivateAttr

from ....core import types as cf
from ....core.analect.analect import AnalectRunContext

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...exceptions import OrchestratorInterruption
from ...tags import Tag, TagLike, unescaped_tag_content

from ..tag_with_id import TagWithIDExtension
from ..tool_use import ToolUseExtension

from .exceptions import InvalidCommandLineInput
from .prompts import (
    COMMAND_LINE_BASH_SCRIPT_BASIC_DESCRIPTION,
    COMMAND_LINE_BASH_SCRIPT_TOOL_USE_DESCRIPTION,
    COMMAND_LINE_BASIC_DESCRIPTION,
)
from .runner import CommandLineInput, CommandLineOutput, run_command_line
from .utils import get_allowed_and_disallowed_commands, get_command_tokens_from_bash
from .validators.cli_command_validator import CliCommandValidator
from .validators.factory import get_default_validators


class CommandLineExtension(TagWithIDExtension, ToolUseExtension):
    name: str = "command_line"
    included_in_system_prompt: bool = True
    trace_tool_execution: bool = False  # Custom trace node will be generated
    tag_name: str = "command_line"
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
    max_output_length: int | None = Field(
        default=None,
        description="Maximum length of the output to include in the response string",
    )
    allow_bash_script: bool = Field(
        default=False,
        description="Whether to allow bash script as input, WARNING: this could have potential security risks since the there will be a weak validation on the input bash script, and should be used with caution",
    )
    enable_tool_use: bool = Field(
        default=False,
        description="Whether to enable tool_use feature provided by the LLM provider, if set to True, bash script will be allowed regardless of allow_bash_script",
    )
    bash_tool: ant.BashTool = Field(
        default_factory=ant.BashTool,
        description="The bash tool to use for tool use feature, only used when enable_tool_use is True",
    )
    cwd: str | None = Field(
        default=None,
        description="Current working directory to run the command in",
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables for the command line execution"
    )
    command_validators: Dict[str, CliCommandValidator] = Field(
        default_factory=lambda: get_default_validators(),
        description="Dictionary of command name to validator for the command",
    )
    _last_command: str | None = PrivateAttr(None)
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

    async def description(self) -> TagLike:
        """
        Return the description of the command line extension.
        """
        desc: list[str | Tag] = [
            COMMAND_LINE_BASH_SCRIPT_TOOL_USE_DESCRIPTION
            if self.enable_tool_use
            else (
                COMMAND_LINE_BASH_SCRIPT_BASIC_DESCRIPTION
                if self.allow_bash_script
                else COMMAND_LINE_BASIC_DESCRIPTION
            ).format(cli_tag_name=self.tag_name)
        ]
        if self.allowed_commands:
            desc.append(
                Tag(
                    name="allowed_commands",
                    contents=[
                        Tag(
                            name="command",
                            attributes={"name": name},
                            contents=desc,
                        )
                        for name, desc in self.allowed_commands.items()
                    ],
                )
            )
        return desc

    def _get_proxy_disabled_commands(self) -> list[str]:
        """Return list of commands that need proxy environment cleaning.

        Override in subclasses to customize proxy handling.
        """
        return []

    def _needs_proxy_cleaning(self, command: str) -> bool:
        """Check if command needs proxy environment cleaning."""
        proxy_commands = self._get_proxy_disabled_commands()
        return any(cmd in command for cmd in proxy_commands)

    def _clean_proxy_env(self, inp: CommandLineInput) -> None:
        """Apply proxy-cleaned environment settings."""
        if inp.env is not None:
            inp.env = {
                k: v
                for k, v in inp.env.items()
                if not any(
                    proxy in k.upper()
                    for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]
                )
            }

    def _get_fresh_environment(self, inp: CommandLineInput) -> None:
        """Get fresh environment variables from subprocess and update inp.env.

        This method gets the current environment state by running 'env -0'
        subprocess command, which provides more up-to-date environment
        variables than os.environ. Custom inp.env values take precedence.

        Args:
            inp: CommandLineInput to update with fresh environment
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

    async def _check_command_specific_validators(
        self,
        inp: CommandLineInput,
        context: AnalectRunContext,
    ) -> CommandLineInput:
        """Validate command using registered validators.

        Args:
            inp: The command input
            context: The run context

        Returns:
            The potentially modified command input
        """
        tokens_for_each_command = get_command_tokens_from_bash(inp.command)
        for command_tokens in tokens_for_each_command:
            if not command_tokens:  # Skip empty commands
                continue

            command = command_tokens[0]
            if command in self.command_validators:
                inp = await self.command_validators[command].run_validator(
                    command_tokens, inp, context
                )

        return inp

    async def on_before_run_command(
        self,
        inp: CommandLineInput,
        context: AnalectRunContext,
    ) -> CommandLineInput:
        """
        This method is called before running a command. It can be used to perform
        any necessary pre-processing or validation before the command execution.
        """
        self._get_fresh_environment(inp)

        # Handle proxy cleaning if needed
        if self._needs_proxy_cleaning(inp.command):
            self._clean_proxy_env(inp)

        return inp

    async def on_run_command_success(
        self,
        inp: CommandLineInput,
        out: CommandLineOutput,
        context: AnalectRunContext,
    ) -> CommandLineOutput:
        """
        This method is called after a command is successfully executed. It can be
        used to perform any necessary post-processing or logging after the command
        execution.
        """
        return out

    async def on_run_command_failed(
        self,
        inp: CommandLineInput,
        exception: Exception,
        context: AnalectRunContext,
    ) -> None:
        """
        This method is called after a command fails to execute. It can be used to
        perform any necessary error handling or logging after the command execution.
        """
        pass

    async def _validate_command(self, command: str) -> str:
        """
        Validate the command before running it.

        Returns:
            The found allowed commands
        """
        # Check against both allowed and disallowed commands
        if not self._tokenized_disallowed_commands:
            # If there are no disallowed commands, use an empty list
            self._tokenized_disallowed_commands = []

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
            # We don't need to use the original command here, but it's available if needed in the future
            raise InvalidCommandLineInput(
                f"`{command}` uses command that is explicitly disallowed: `{disallowed_cmd}`. Reason: {reason}"
            )

        # Then check for commands that aren't in the allowed list
        if result.disallowed:
            raise InvalidCommandLineInput(
                f"`{command}` uses commands that aren't allowed: `{'`,`'.join(result.disallowed)}`. Please use only allowed commands: {','.join(self.allowed_commands.keys())}"
            )

        return ",".join(result.allowed)

    async def on_command(
        self,
        identifier: str,
        command: str,
        context: AnalectRunContext,
        cwd: str | None = None,
        attrs: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandLineOutput:
        await context.io.system(
            f"Validating command `{command}`", run_label="Validating command"
        )
        name = ""
        try:
            name = await self._validate_command(command=command)
            inp = CommandLineInput(
                identifier=identifier,
                command=command,
                cwd=cwd or self.cwd,
                max_output_lines=self.max_output_lines,
                max_output_length=self.max_output_length,
                attrs=attrs,
                env=env or self.env,
            )
            inp = await self._check_command_specific_validators(inp, context)
        except InvalidCommandLineInput as e:
            await context.io.system(
                str(e), run_label="Validation failed", run_status=cf.RunStatus.FAILED
            )
            raise e

        inp = await self.on_before_run_command(inp=inp, context=context)
        await context.io.system(
            dedent(
                """\
                Running command in `{cwd}`:
                ```console
                {command}
                ```
                """
            ).format(command=inp.command, cwd=inp.cwd or ""),
            run_label=f"Running command {name}",
        )
        runnable = RunnableLambda(run_command_line, name=name)
        try:
            out = await context.invoke(runnable, inp, run_type="tool")
        except Exception as e:
            msg = f"Command `{name}` failed due to {type(e).__name__}: {e}"
            await context.io.system(
                msg, run_label=f"Command {name} failed", run_status=cf.RunStatus.FAILED
            )
            await self.on_run_command_failed(inp=inp, exception=e, context=context)
            raise e

        await context.io.system(
            out.to_markdown(),
            run_label=f"Command {name} succeeded"
            if out.success
            else f"Command {name} failed",
            run_status=cf.RunStatus.COMPLETED if out.success else cf.RunStatus.FAILED,
        )

        out = await self.on_run_command_success(inp=inp, out=out, context=context)

        return out

    async def on_command_tag(
        self,
        identifier: str,
        content: str,
        context: AnalectRunContext,
        cwd: str | None = None,
        attrs: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        try:
            out = await self.on_command(
                identifier=identifier,
                command=content,
                context=context,
                cwd=cwd,
                attrs=attrs,
                env=env,
            )
        except Exception as e:
            raise OrchestratorInterruption(
                f"Command failed due to {type(e).__name__}: {e}"
            )

        raise OrchestratorInterruption(out.to_tag().prettify())

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        if tag.name != self.tag_name:
            return

        try:
            identifier: str = (
                tag.get("identifier", self.default_identifier)
                if self.default_identifier
                else tag["identifier"]
            )
            content: str = unescaped_tag_content(tag)
            attrs = tag.attrs or {}
            cwd = attrs.get("cwd", None)
            await self.on_command_tag(
                identifier=identifier,
                content=content,
                context=context,
                cwd=cwd,
                attrs=attrs,
                env=self.env,
            )

        except KeyError as exc:
            raise OrchestratorInterruption(
                f"Invalid tag, missing required attribute: {exc}"
            )

    @property
    async def tools(self) -> list[ant.ToolLike]:
        if self.enable_tool_use:
            return [self.bash_tool]

        return []

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        bash_input = ant.BashInput.parse_obj(tool_use.input)
        command = self._get_bash_input_command(bash_input)
        out = await self.on_command(
            identifier=tool_use.id,
            command=command,
            context=context,
            env=self.env,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content=out.to_tag().prettify(),
            is_error=not out.success,
        )

    def _get_bash_input_command(self, bash_input: ant.BashInput) -> str:
        """
        Get the command from the bash input.
        """
        if bash_input.restart:
            if self._last_command is None:
                raise RuntimeError(
                    "Cannot restart bash when there is no previous command"
                )

            return self._last_command

        if bash_input.command is None:
            raise RuntimeError("`command` is required when `restart` is not true")

        return bash_input.command
