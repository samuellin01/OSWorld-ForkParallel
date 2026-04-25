# pyre-strict


from typing import Any

from google.genai import errors, types
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from ....utils.decorators import retryable, RETRYABLE_CONNECTION_ERRS

from ..bedrock.api.invoke_model import anthropic as ant

from ..bedrock.exceptions import UnexpectedEmptyResponseException
from ..bedrock.utils import lc_message_to_ant_message, lc_message_to_ant_system

from .base import GoogleBase
from .model import get_model
from .utils import (
    ant_message_to_google,
    ant_system_to_google,
    ant_thinking_to_thinking_config,
    ant_tool_choice_to_tool_config,
    ant_tools_to_google,
    generate_content_response_to_ant_response,
)

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = RETRYABLE_CONNECTION_ERRS + (
    errors.ServerError,
    UnexpectedEmptyResponseException,
)


class GeminiChat(GoogleBase, BaseChatModel):
    """LangChain Wrapper around Google Gemini chat language model."""

    # The following params is for compatibility with Anthropic API, the current anchor for LLM APIs. We will replace this with a unified API in the future.
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
        return "gemini-chat"

    @property
    def _is_thinking_enabled(self) -> bool:
        """Return whether the model supports thinking."""
        return (
            self.thinking is not None and self.thinking.type == ant.ThinkingType.ENABLED
        )

    def _get_generate_content_config(
        self, input_messages: list[BaseMessage]
    ) -> types.GenerateContentConfig:
        """Get the generate content config."""
        system = None
        for msg in input_messages:
            if isinstance(msg, SystemMessage):
                system = lc_message_to_ant_system(msg)
                break

        # pyre-fixme[28]: Unexpected keyword argument `system_instruction` to GenerateContentConfig
        return types.GenerateContentConfig(
            system_instruction=ant_system_to_google(system),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            frequency_penalty=self.frequency_penalty,
            tools=ant_tools_to_google(self.tools) if self.tools else None,
            max_output_tokens=self.max_tokens,
            tool_config=(
                ant_tool_choice_to_tool_config(self.tool_choice)
                if self.tool_choice
                else None
            ),
            thinking_config=(
                ant_thinking_to_thinking_config(self.thinking)
                if self.thinking is not None
                else None
            ),
        )

    async def _get_contents(
        self,
        messages: list[BaseMessage],
    ) -> Any:
        """Get the contents for generate content request."""
        ant_messages: list[ant.Message] = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                ant_messages.append(await lc_message_to_ant_message(msg))
        return [ant_message_to_google(am) for am in ant_messages]

    async def _invoke_model(
        self,
        contents: types.ContentListUnion,  # pyre-fixme[11]: Annotation `types.ContentListUnion` is not defined as a type
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Invoke the model."""
        response = await self.client.aio.models.generate_content(
            model=get_model(self.model),
            contents=contents,
            config=config,
        )

        if not response.candidates:
            raise UnexpectedEmptyResponseException("No candidates found in response")

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

        @retryable(exceptions=RETRYABLE_EXCEPTIONS, **self.retryable_config.dict())
        async def _invoke_model_with_retry(
            contents: types.ContentListUnion,
            config: types.GenerateContentConfig,
        ) -> types.GenerateContentResponse:
            return await self._invoke_model(contents=contents, config=config)

        gg_response = await _invoke_model_with_retry(
            contents=await self._get_contents(messages),
            config=self._get_generate_content_config(messages),
        )

        response = generate_content_response_to_ant_response(gg_response)

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
