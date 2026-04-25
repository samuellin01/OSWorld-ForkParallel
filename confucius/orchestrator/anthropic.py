# pyre-strict

import asyncio
import re
from typing import override

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field, PrivateAttr

from ..core import types as cf
from ..core.analect import AnalectRunContext
from ..core.chat_models.azure.openai import OpenAIChat
from ..core.chat_models.bedrock.anthropic import ClaudeChat

from ..core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..core.chat_models.google.gemini import GeminiChat

from ..core.llm_manager import LLMParams
from ..core.memory import CfMessage
from .exceptions import OrchestratorInterruption
from .extensions import ToolUseExtension, ToolUseObserver
from .llm import LLMOrchestrator

TOOL_USE_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
NON_TOOL_USE_NAME_PATTERN: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9_-]")
TOKEN_EFFICIENT_BETA_TAG: str = "token-efficient-tools-2025-02-19"
INTERLEAVED_THINKING_TAG: str = "interleaved-thinking-2025-05-14"


class AnthropicLLMOrchestrator(LLMOrchestrator):
    """
    Orchestrator that is compatible with the Anthropic API and also backward compatible with metagen APIs
    """

    tool_choice: ant.ToolChoice | None = Field(
        default=None,
        description="The tool choice to use for the LLM",
    )
    # A queue storing the unprocessed tool use messages
    _tool_use_queue: list[ant.MessageContentToolUse] = PrivateAttr([])

    async def get_llm_params(self) -> LLMParams:
        llm_params = (await super().get_llm_params()).copy()

        if tools := (await self.tools):
            if llm_params.additional_kwargs is None:
                llm_params.additional_kwargs = {}

            # Add tools to additional_kwargs
            llm_params.additional_kwargs["tools"] = tools

            if (
                llm_params.model is not None
                and "claude-3-7-sonnet-20250219" in llm_params.model
                and (
                    not (
                        self.tool_choice is not None
                        and self.tool_choice.disable_parallel_tool_use
                    )
                )
            ):
                # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use
                # Only 3.7 model supports token-efficient tool use
                llm_params = self._add_beta_tag(llm_params, TOKEN_EFFICIENT_BETA_TAG)

            if llm_params.model is not None and (
                "claude-sonnet-4" in llm_params.model
                or "claude-opus-4" in llm_params.model
                or "claude-4" in llm_params.model
            ):
                # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking
                # Only 4.0 model supports interleaved thinking
                llm_params = self._add_beta_tag(llm_params, INTERLEAVED_THINKING_TAG)

            if self.tool_choice is not None:
                assert llm_params.additional_kwargs is not None
                llm_params.additional_kwargs["tool_choice"] = self.tool_choice
        return llm_params

    def _add_beta_tag(self, llm_params: LLMParams, tag: str) -> LLMParams:
        if llm_params.additional_kwargs is None:
            llm_params.additional_kwargs = {}

        additional_kwargs = dict(llm_params.additional_kwargs)
        if "beta" in additional_kwargs:
            beta_list = additional_kwargs["beta"]
            # Make sure beta_list is a list
            if not isinstance(beta_list, list):
                beta_list = [beta_list]

            # Add our tag if it's not already there
            if tag not in beta_list:
                beta_list.append(tag)
            additional_kwargs["beta"] = beta_list
        else:
            additional_kwargs["beta"] = [tag]

        llm_params.additional_kwargs = additional_kwargs
        return llm_params

    async def _invoke_llm_impl(
        self,
        chat: BaseChatModel,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> str:
        if isinstance(chat, (ClaudeChat, GeminiChat, OpenAIChat)):
            return await self._invoke_claude_impl(chat, messages, context)
        else:
            return await super()._invoke_llm_impl(chat, messages, context)

    async def _invoke_claude_impl(
        self,
        chat: BaseChatModel,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> str:
        response = await context.invoke(chat, messages)
        response = await self.on_llm_response(response, context)
        return await self._process_response(response, context)

    async def on_thinking(
        self, content: ant.MessageContentThinking, context: AnalectRunContext
    ) -> None:
        await context.io.system(
            content.thinking,
            run_status=cf.RunStatus.COMPLETED,
            run_label="Thinking",
        )
        context.memory_manager.add_messages(
            [CfMessage(type=cf.MessageType.AI, content=[content.dict()])]
        )

    async def on_redacted_thinking(
        self, content: ant.MessageContentRedactedThinking, context: AnalectRunContext
    ) -> None:
        await context.io.system(
            f"{content.data}\n(Contents redacted)",
            run_status=cf.RunStatus.COMPLETED,
            run_label="Thinking",
            run_description="Content redacted",
        )
        context.memory_manager.add_messages(
            [CfMessage(type=cf.MessageType.AI, content=[content.dict()])]
        )

    @override
    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """
        Callback after processing the messages is complete. Override the default behavior from base orchestrator to do nothing.
        For Anthropic API, we want to process the tool use queue after all messages are processed before we consider the messages processed.

        Args:
            context (AnalectRunContext): The context of the collector.

        Raise an OrchestratorInterruption to prompt the orchestrator to continue
        """
        pass

    async def _on_process_tool_use_queue_complete(
        self, context: AnalectRunContext
    ) -> None:
        """
        Callback after processing the tool use queue is complete. Default behavior is sequentially call each extension's on_process_messages_complete.

        Args:
            context (AnalectRunContext): The context of the collector.

        Raise an OrchestratorInterruption to prompt the orchestrator to continue
        """
        await super().on_process_messages_complete(context)

    async def _process_response(
        self, msg: BaseMessage, context: AnalectRunContext
    ) -> str:
        """
        Post process the response from the LLM and return the text response for other extensions to process.
        """

        response = ant.Response.parse_obj(msg.response_metadata)
        text_responses = []
        for ct in response.content:
            if isinstance(ct, ant.MessageContentText):
                text_responses.append(ct.text)
            elif isinstance(ct, ant.MessageContentThinking):
                await self.on_thinking(ct, context)
            elif isinstance(ct, ant.MessageContentRedactedThinking):
                await self.on_redacted_thinking(ct, context)
            elif isinstance(ct, ant.MessageContentToolUse):
                self._tool_use_queue.append(self._fix_tool_use(ct))

        return "\n".join(text_responses)

    async def _process_messages(self, task: str, context: AnalectRunContext) -> None:
        await super()._process_messages(task, context)

        if self._tool_use_queue:
            try:
                await self._process_tool_use_queue(context)
            except OrchestratorInterruption as exc:
                await self._process_interruption(exc, context)
            finally:
                self._tool_use_queue.clear()

            await self._process_messages(task, context)
        else:
            # Call the _on_process_tool_use_queue_complete hook which will raise an OrchestratorInterruption
            # if the orcehstrator should be triggered again
            # We do a try/except style here to match the behavior of the base orchestrator
            try:
                await self._on_process_tool_use_queue_complete(context)
            except OrchestratorInterruption as exc:
                await self._process_interruption(exc, context)
                await self._process_messages(task, context)

    @property
    def _enabled_tool_use_extensions(self) -> list[ToolUseExtension]:
        return [
            ext
            for ext in self.extensions
            if isinstance(ext, ToolUseExtension) and ext.enable_tool_use
        ]

    @property
    def _tool_use_observers(self) -> list[ToolUseObserver]:
        return [ext for ext in self.extensions if isinstance(ext, ToolUseObserver)]

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            tool
            for ext in self._enabled_tool_use_extensions
            for tool in await ext.tools
        ]

    @property
    async def _all_tool_names(self) -> set[str]:
        res = set()
        for ext in self._enabled_tool_use_extensions:
            res.update(await ext.all_tool_names)
        return res

    def _fix_tool_use(
        self, tool_use: ant.MessageContentToolUse
    ) -> ant.MessageContentToolUse:
        """
        Claude API requires the tool name match pattern '^[a-zA-Z0-9_-]{1,64}$', otherwise it will throw an error.
        This function sanitizes tool names to comply with this requirement.
        """
        name = tool_use.name
        if not TOOL_USE_NAME_PATTERN.match(name):
            # Create a copy to avoid modifying the original object
            tool_use = tool_use.copy()
            # Replace invalid characters with underscores
            tool_use.name = NON_TOOL_USE_NAME_PATTERN.sub("_", name)
            # Truncate if longer than 64 characters
            if len(tool_use.name) > 64:
                tool_use.name = tool_use.name[:64]
        return tool_use

    async def _process_tool_use_queue(self, context: AnalectRunContext) -> None:
        all_tool_names = await self._all_tool_names

        for tool_use in self._tool_use_queue:
            for ext in self._tool_use_observers:
                await ext.on_before_tool_use(tool_use=tool_use, context=context)

            tool_result = await self._process_tool_use(
                tool_use, all_tool_names, context
            )

            if tool_result is None:
                continue

            for ext in self._tool_use_observers:
                await ext.on_after_tool_use_result(
                    tool_use=tool_use, tool_result=tool_result, context=context
                )

    async def _process_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        all_tool_names: set[str],
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult | None:
        context.memory_manager.add_messages(
            [CfMessage(type=cf.MessageType.AI, content=[tool_use.dict()])]
        )

        if tool_use.name not in all_tool_names:
            tool_result = ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Tool `{tool_use.name}` is not supported, please try another tool",
                is_error=True,
            )
            context.memory_manager.add_messages(
                [
                    CfMessage(
                        type=cf.MessageType.HUMAN,
                        content=[tool_result.dict()],
                    )
                ]
            )
            return tool_result

        final_tool_result = None
        for ext in self._enabled_tool_use_extensions:
            await ext.on_before_tool_use(tool_use=tool_use, context=context)
            tool_result = await ext._on_tool_use(tool_use=tool_use, context=context)
            if tool_result is None:
                continue

            context.memory_manager.add_messages(
                [CfMessage(type=cf.MessageType.HUMAN, content=[tool_result.dict()])]
            )
            await ext.on_after_tool_use_result(
                tool_use=tool_use, tool_result=tool_result, context=context
            )
            final_tool_result = tool_result

        return final_tool_result
