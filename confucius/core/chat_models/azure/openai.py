# pyre-strict

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from openai import BadRequestError, RateLimitError
from pydantic import Field

from ....utils.decorators import retryable, RETRYABLE_CONNECTION_ERRS

from ..bedrock.api.invoke_model import anthropic as ant

from .adapters.chat_completions import ChatCompletionsAdapter
from .adapters.responses import ResponsesAPIAdapter

from .base import AzureBase, OpenAIBase

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = RETRYABLE_CONNECTION_ERRS + (
    RateLimitError,
    BadRequestError,
)


class OpenAIChat(AzureBase, BaseChatModel):
    """OpenAI Chat model with adapter pattern for API selection.

    Supports both chat.completions and responses APIs through specialized adapters.
    """

    # API Selection
    use_responses_api: bool = Field(
        default=True,
        description="Whether to use responses API (True) or chat.completions API (False)",
    )

    # The following params is for compatibility with Anthropic API
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
    def _adapter(self) -> OpenAIBase:
        """Get the appropriate adapter instance based on use_responses_api flag."""
        # Create adapter configuration by extracting relevant fields
        adapter_config = {
            "client": self.client,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_k": getattr(self, "top_k", None),
            "top_p": getattr(self, "top_p", None),
            "frequency_penalty": getattr(self, "frequency_penalty", None),
        }

        if self.use_responses_api:
            return ResponsesAPIAdapter(**adapter_config)
        else:
            return ChatCompletionsAdapter(**adapter_config)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "Sync version _generate is not recommended for OpenAIChat"
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call Azure OpenAI to generate chat completions using the selected adapter."""

        @retryable(exceptions=RETRYABLE_EXCEPTIONS, **self.retryable_config.dict())
        async def _generate_with_retry() -> ant.Response:
            return await self._adapter.generate(messages, **kwargs)

        # Get the response using the adapter
        response = await _generate_with_retry()

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
