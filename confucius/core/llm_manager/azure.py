# pyre-strict

import logging
import os

from langchain_core.language_models import BaseChatModel
from openai import AsyncAzureOpenAI
from pydantic import PrivateAttr

from ..chat_models.azure.openai import OpenAIChat

from .base import LLMManager, LLMParams
from .constants import DEFAULT_INITIAL_MAX_TOKEN

logger: logging.Logger = logging.getLogger(__name__)


class AzureLLMManager(LLMManager):
    """Azure OpenAI manager using native AsyncAzureOpenAI client only.

    Environment variables used:
    - AZURE_OPENAI_ENDPOINT: e.g. https://<resource>.openai.azure.com/
    - AZURE_OPENAI_API_KEY: API key
    - AZURE_OPENAI_API_VERSION: e.g. 2024-10-21 (optional, default 2024-10-21)
    - AZURE_OPENAI_DEPLOYMENT: deployment name to use when params.model is not provided
    """

    _client: AsyncAzureOpenAI | None = PrivateAttr(default=None)

    def get_client(self) -> AsyncAzureOpenAI:
        """Create/cache AsyncAzureOpenAI client configured via env vars."""
        if self._client is None:
            endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
            # AsyncAzureOpenAI supports azure_deployment on the client, but we pass model per call via OpenAIChat
            self._client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        return self._client

    def _get_chat(self, params: LLMParams) -> BaseChatModel:
        """Get Azure OpenAI chat model using native client configured by env."""
        model = params.model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        if not model:
            raise ValueError(
                "Azure model/deployment not specified. Set params.model or AZURE_OPENAI_DEPLOYMENT env var."
            )

        return OpenAIChat(
            client=self.get_client(),
            model=model,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=(
                params.max_tokens
                or params.initial_max_tokens
                or DEFAULT_INITIAL_MAX_TOKEN
            ),
            stop=params.stop,
            cache=params.cache,
            **(params.additional_kwargs or {}),
            use_responses_api=True,
        )
