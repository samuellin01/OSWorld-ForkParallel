# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, List
from unittest.mock import Mock

from langchain_core.messages import BaseMessage
from openai import AsyncAzureOpenAI
from pydantic import Field

from ..base_chat import ConfuciusBaseChat

from ..bedrock.api.invoke_model import anthropic as ant


class AzureBase(ConfuciusBaseChat):
    client: AsyncAzureOpenAI | Mock = Field(
        ..., description="The Azure client to use for the Azure OpenAI API"
    )

    top_k: float | None = Field(
        default=None,
        description="For each token selection step, the ``top_k`` tokens with the highest probabilities are sampled.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Positive values penalize tokens that repeatedly appear in the generated text, increasing the probability of generating more diverse content.",
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azure"


class OpenAIBase(ABC):
    """Lightweight base class for OpenAI adapters - just shared configuration and interface."""

    def __init__(
        self,
        client: AsyncAzureOpenAI | Mock,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: ant.Thinking | None = None,
        tool_choice: ant.ToolChoice | None = None,
        tools: list[ant.ToolLike] | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.tool_choice = tool_choice
        self.tools = tools
        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty

    @abstractmethod
    async def _invoke_api(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        """Invoke the specific OpenAI API endpoint.

        Subclasses implement this to call their specific API (chat.completions or responses).

        Args:
            messages: List of messages to send to the API
            **kwargs: Additional parameters specific to the API

        Returns:
            Raw API response object
        """
        pass

    @abstractmethod
    def _convert_response(self, raw_response: Any) -> ant.Response:
        """Convert raw API response to Anthropic response format.

        Subclasses implement this to handle their specific response format.

        Args:
            raw_response: Raw response from the API

        Returns:
            Converted response in Anthropic format
        """
        pass

    async def generate(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> ant.Response:
        """Main generation method used by both adapters.

        This method orchestrates the API call and response conversion.

        Args:
            messages: List of messages to process
            **kwargs: Additional parameters

        Returns:
            Response in Anthropic format
        """
        raw_response = await self._invoke_api(messages, **kwargs)
        return self._convert_response(raw_response)
