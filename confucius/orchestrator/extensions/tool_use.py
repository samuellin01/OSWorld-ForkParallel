# pyre-strict

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from langchain_core.runnables import RunnableLambda
from pydantic import Field

from ...core.analect import AnalectRunContext

from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...core.memory import CfMessage
from ..extensions.base import Extension
from ..tags import Tag


class ToolUseExtension(Extension, ABC):
    exceptions: tuple[type[Exception], ...] = Field(
        default=(Exception,),
        description="Exceptions to retry when the runnable fails, which will interrupt the orchestrator but not terminate",
    )
    is_retryable_ex: Callable[[Exception], bool] | None = Field(
        None,
        description="Function to determine if an exception is retryable, if provided, overrides exceptions both True or False",
    )
    enable_tool_use: bool = Field(
        default=True,
        description="Whether to enable tool_use feature provided by the LLM provider",
    )
    trace_tool_execution: bool = Field(
        default=True,
        description="Whether to trace tool execution as a node in the trace, if not enabled, the tool use will be hidden in the trace, and you can write your own trace node to show the tool use",
    )
    run_type: str | None = Field(
        "tool",
        description="The run type of the runnable, only used when enabled_tracing is True",
    )

    def get_non_retryable_exceptions_message(
        self, tool_use_id: str, exc: BaseException
    ) -> CfMessage:
        if isinstance(exc, asyncio.CancelledError):
            if self.enable_tool_use:
                return CfMessage(
                    content=[
                        ant.MessageContentToolResult(
                            tool_use_id=tool_use_id,
                            content=f"tool_use (id: {tool_use_id}) was cancelled by user",
                            is_error=True,
                        ).dict()
                    ]
                )
            else:
                return CfMessage(
                    content=Tag(
                        name="exception",
                        contents="Cancelled by user",
                        attributes={"identifier": tool_use_id, "name": self.name},
                    ).prettify()
                )

        if self.enable_tool_use:
            return CfMessage(
                content=[
                    ant.MessageContentToolResult(
                        tool_use_id=tool_use_id,
                        content=f"Unexpected error when processing tool use (id: {tool_use_id}): {type(exc).__name__}: {str(exc)}",
                        is_error=True,
                    ).dict()
                ]
            )
        else:
            return CfMessage(
                content=Tag(
                    name="exception",
                    contents=f"Unexpected error: {type(exc).__name__}: {str(exc)}",
                    attributes={"identifier": tool_use_id, "name": self.name},
                ).prettify()
            )

    async def _on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult | None:
        """
        Actual implementation of tool use handling including error handling and interruption handling.

        Args:
            tool_use: The tool use content containing the tool name and parameters
            context: The current execution context providing access to memory and other resources

        Returns:
            ant.MessageContentToolResult: A result object containing the tool execution output, or None if the tool is not supported
        """
        if not self.enable_tool_use:
            raise ValueError(f"Tool use is not enabled for {self.__class__.__name__}")

        if tool_use.name not in (await self.all_tool_names):
            return None

        try:
            return await self._on_tool_use_impl(tool_use, context)
        except BaseException as exc:
            context.memory_manager.add_messages(
                [self.get_non_retryable_exceptions_message(tool_use.id, exc)]
            )
            raise

    @property
    async def all_tool_names(self) -> set[str]:
        return {tool.name for tool in (await self.tools)}

    async def _on_tool_use_impl(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        try:
            if self.trace_tool_execution:

                async def tool_use_wrapper(inp: dict[str, Any]) -> dict[str, Any]:
                    result = await self.on_tool_use(
                        tool_use=tool_use,
                        context=context,
                    )
                    return result.model_dump()

                runnable = RunnableLambda(tool_use_wrapper, name=tool_use.name)
                return ant.MessageContentToolResult.parse_obj(
                    await context.invoke(
                        runnable, tool_use.input, run_type=self.run_type
                    )
                )

            else:
                result = await self.on_tool_use(
                    tool_use=tool_use,
                    context=context,
                )

        except Exception as exc:
            is_retryable = isinstance(exc, tuple(self.exceptions))
            if self.is_retryable_ex:
                is_retryable = self.is_retryable_ex(exc)

            if is_retryable:
                return ant.MessageContentToolResult(
                    tool_use_id=tool_use.id,
                    content=f"tool use `{tool_use.name}` failed due to: {type(exc).__name__}: {str(exc)}",
                    is_error=True,
                )
            else:
                raise

        return result

    @abstractmethod
    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """
        Abstract method that handles tool usage requests.

        This method is called when a tool use tag is encountered in the content.
        Implementations should process the tool request and return the result.

        Args:
            tool_use: The tool use content containing the tool name and parameters
            context: The current execution context providing access to memory and other resources

        Returns:
            ant.MessageContentToolResult: A result object containing the tool execution output
        """
        ...

    @property
    @abstractmethod
    async def tools(self) -> list[ant.ToolLike]:
        """
        Property to define the tools for the LLM.

        Returns:
            list[ant.ToolLike]: A list of tools that the LLM should use. The schema of the tool depends on the LLM provider.
        """

        ...

    async def on_before_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> None:
        """
        Callback to process the tool use before it is executed.

        Args:
            tool_use: The tool use content containing the tool name and parameters
            context: The current execution context providing access to memory and other resources
        """
        pass

    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        """
        Callback to process the tool use result after it is executed.

        You can use this to do some post-processing on the tool use result, or append user messages to the memory.
        More information: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use

        Args:
            tool_use: The tool use content containing the tool name and parameters
            tool_result: The result of the tool use execution
            context: The current execution context providing access to memory and other resources
        """
        pass


class ToolUseObserver(Extension):
    """
    An observer that monitors tool uses and tool results and reacts to execution events.
    """

    name: str = "tool_use_observer"
    included_in_system_prompt: bool = False

    async def on_before_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> None:
        """
        Callback to process the tool use before it is executed.

        Args:
            tool_use: The tool use content containing the tool name and parameters
            context: The current execution context providing access to memory and other resources
        """
        pass

    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        """
        Callback to process the tool use result after it is executed.

        You can use this to do some post-processing on the tool use result.

        Args:
            tool_use: The tool use content containing the tool name and parameters
            tool_result: The result of the tool use execution
            context: The current execution context providing access to memory and other resources
        """
        pass
