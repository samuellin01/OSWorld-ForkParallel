# pyre-strict

import atexit
import json
import logging
from typing import Any, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from ....utils.asyncio import convert_to_async
from ....utils.decorators import retryable, RETRYABLE_CONNECTION_ERRS

from .api.invoke_model import anthropic as ant

from .base import BedrockBase
from .exceptions import (
    bedrock_exception_handling,
    BedrockInvalidResponseException,
    BedrockModelErrorException,
    BedrockServiceUnavailableException,
    BedrockThrottlingException,
    BedrockValidationException,
    UnexpectedEmptyResponseException,
)
from .model_id import get_model_id
from .utils import (
    append_stop_sequence,
    lc_message_to_ant_message,
    lc_message_to_ant_system,
)

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = RETRYABLE_CONNECTION_ERRS + (
    BedrockServiceUnavailableException,
    BedrockThrottlingException,
    BedrockModelErrorException,
    UnexpectedEmptyResponseException,
)


logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level token usage collector
# ---------------------------------------------------------------------------
# Accumulates raw API usage from every _agenerate call during the process
# lifetime.  Written to ``token_usage.json`` at process exit via atexit.


class _TokenUsageCollector:
    """Collects per-call token usage and dumps to token_usage.json on exit."""

    def __init__(self) -> None:
        self._input_tokens: List[int] = []
        self._output_tokens: List[int] = []
        self._uncached_input_tokens: List[int] = []
        self._cache_creation_input_tokens: List[int] = []
        self._cache_read_input_tokens: List[int] = []

    def record(self, usage: ant.Usage) -> None:
        total_input = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        self._input_tokens.append(total_input)
        self._output_tokens.append(usage.output_tokens)
        self._uncached_input_tokens.append(usage.input_tokens)
        self._cache_creation_input_tokens.append(usage.cache_creation_input_tokens or 0)
        self._cache_read_input_tokens.append(usage.cache_read_input_tokens or 0)

    def dump(self) -> None:
        """Write token_usage.json if any calls were recorded."""
        if not self._input_tokens:
            return
        data = {
            "num_turns": len(self._input_tokens),
            "input_tokens_per_turn": self._input_tokens,
            "output_tokens_per_turn": self._output_tokens,
            "uncached_input_tokens_per_turn": self._uncached_input_tokens,
            "cache_creation_input_tokens_per_turn": self._cache_creation_input_tokens,
            "cache_read_input_tokens_per_turn": self._cache_read_input_tokens,
            "total_input": sum(self._input_tokens),
            "total_output": sum(self._output_tokens),
            "total_uncached_input": sum(self._uncached_input_tokens),
            "total_cache_creation_input": sum(self._cache_creation_input_tokens),
            "total_cache_read_input": sum(self._cache_read_input_tokens),
        }
        try:
            with open("token_usage.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write token_usage.json: %s", e)


_token_collector = _TokenUsageCollector()
atexit.register(_token_collector.dump)


class ClaudeChat(BedrockBase, BaseChatModel):
    """LangChain Wrapper around Bedrock Claude chat language model."""

    thinking: ant.Thinking | None = Field(default=None)
    tool_choice: ant.ToolChoice | None = Field(default=None)
    tools: list[ant.ToolLike] | None = Field(default=None)

    include_stop_sequence: bool = Field(
        default=True,
        description="Whether to include the stop sequence in the response.",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "claude-chat"

    @property
    def _is_thinking_enabled(self) -> bool:
        """Return whether the model supports thinking."""
        return (
            self.thinking is not None and self.thinking.type == ant.ThinkingType.ENABLED
        )

    async def _get_invoke_model_request_body(
        self, input_messages: list[BaseMessage]
    ) -> ant.Body:
        """Get the request body for the invoke_model request."""
        effective_temperature = self.temperature
        effective_top_p = self.top_p
        if self._is_thinking_enabled:
            effective_temperature = 1.0
            effective_top_p = None
        elif self.temperature is not None and self.top_p is not None:
            logger.warning(
                "Both `temperature` and `top_p` are set; "
                "`top_p` will be ignored because the model does not accept both simultaneously."
            )
            effective_top_p = None

        body = ant.Body(
            anthropic_version=self.version,
            anthropic_beta=self.beta,
            stop_sequences=self.stop,
            temperature=effective_temperature,
            top_p=effective_top_p,
            max_tokens=self.max_tokens or 8192,
            thinking=self.thinking,
            tool_choice=self.tool_choice,
            tools=self.tools,
        )
        for idx, msg in enumerate(input_messages):
            last_role: ant.MessageRole | None = (
                body.messages[-1].role if body.messages else None
            )
            if idx == 0 and isinstance(msg, SystemMessage):
                body.system = lc_message_to_ant_system(msg)

            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                ant_msg = await lc_message_to_ant_message(msg)
                # Anthropic API does not support DEVELOPER role for now
                if ant_msg.role == ant.MessageRole.DEVELOPER:
                    ant_msg.role = ant.MessageRole.USER

                if last_role == ant_msg.role:
                    body.messages[-1].content.extend(ant_msg.content)
                else:
                    body.messages.append(ant_msg)

        return body

    def _get_serialized_body(self, body: ant.Body) -> str:
        """Serialize the body for the invoke_model request."""
        return body.model_dump_json(exclude_none=True)

    def _parse_response_body(self, response_body: str) -> ant.Response:
        """Parse the response from the invoke_model request."""
        return ant.Response.parse_raw(response_body)

    async def _invoke_model(self, body: ant.Body) -> ant.Response:
        """Invoke the model and return the response."""
        with bedrock_exception_handling():
            modelId = get_model_id(self.model)
            logger.info(f"Sending bedrock request to {modelId}")
            raw_response = await convert_to_async(
                self.client.invoke_model,
                modelId=modelId,
                body=self._get_serialized_body(body),
            )

        try:
            response = self._parse_response_body(raw_response["body"].read())
        except Exception as e:
            raise BedrockInvalidResponseException(
                f"Failed to parse response: {e}",
                metadata=raw_response.get("ResponseMetadata"),
            ) from e

        response.metadata = raw_response.get("ResponseMetadata")
        if response.stop_reason == ant.StopReason.END_TURN and not response.content:
            # when END_TURN is the stop reason, the response can not be empty
            raise UnexpectedEmptyResponseException(
                "Empty response with END_TURN stop reason", metadata=response.metadata
            )
        return response

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "Sync version _generate is not recommended for ClaudeChat"
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call MetaGen to generate chat completions. async version"""
        body = await self._get_invoke_model_request_body(messages)

        @retryable(exceptions=RETRYABLE_EXCEPTIONS, **self.retryable_config.dict())
        async def _invoke_model_with_retry(body: ant.Body) -> ant.Response:
            return await self._invoke_model(body)

        response = await _invoke_model_with_retry(body)

        # Record token usage from the raw API response
        _token_collector.record(response.usage)

        if self.include_stop_sequence:
            response = append_stop_sequence(response)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=[ct.dict(exclude_none=True) for ct in response.content],
                        response_metadata=response.dict(exclude_none=True),
                    ),
                )
            ]
        )
