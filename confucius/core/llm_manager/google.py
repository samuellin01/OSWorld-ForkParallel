# pyre-strict

import logging
import os

from google import genai
from langchain_core.language_models import BaseChatModel
from pydantic import PrivateAttr

from ..chat_models.google.gemini import GeminiChat

from .base import LLMManager, LLMParams
from .constants import DEFAULT_INITIAL_MAX_TOKEN


logger: logging.Logger = logging.getLogger(__name__)


class GoogleLLMManager(LLMManager):
    """Google Gemini manager using native google-genai SDK only.

    Supports both Gemini Developer API and Vertex AI via env configuration.

    Environment variables:
    - GOOGLE_API_KEY or GEMINI_API_KEY: for Developer API
    - GOOGLE_GENAI_USE_VERTEXAI=true: enable Vertex AI mode
      plus GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
    """

    _client: genai.Client | None = PrivateAttr(default=None)

    def get_client(self) -> genai.Client:
        if self._client is None:
            use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {
                "1",
                "true",
                "yes",
            }
            if use_vertex:
                project = os.environ["GOOGLE_CLOUD_PROJECT"]
                location = os.environ["GOOGLE_CLOUD_LOCATION"]
                self._client = genai.Client(
                    vertexai=True, project=project, location=location
                )
            else:
                # API key picked automatically from GOOGLE_API_KEY or GEMINI_API_KEY
                self._client = genai.Client()
        return self._client

    def _get_chat(self, params: LLMParams) -> BaseChatModel:
        model = params.model or os.environ.get("GEMINI_MODEL", "")
        if not model:
            raise ValueError(
                "Gemini model not specified. Set params.model or GEMINI_MODEL env var."
            )
        if "gemini" in model.lower():
            return GeminiChat(
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
            )
        else:
            raise ValueError(
                f"Model: {params.model} is not supported by Google LLM Manager"
            )
