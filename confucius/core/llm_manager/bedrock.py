# pyre-strict

import logging
import os

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from langchain_core.language_models import BaseChatModel
from pydantic import PrivateAttr

from ..chat_models.bedrock.anthropic import ClaudeChat

from .base import LLMManager, LLMParams
from .constants import DEFAULT_INITIAL_MAX_TOKEN


logger: logging.Logger = logging.getLogger(__name__)


class BedrockLLMManager(LLMManager):
    """AWS Bedrock manager using native boto3 client only.

    Environment variables used:
    - AWS_REGION: e.g. us-east-1
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (optional) AWS_SESSION_TOKEN
    - BEDROCK_MODEL_ID: default model id when params.model is not provided
    - ANTHROPIC_VERSION: must be "bedrock-2023-05-31" for Claude Messages API (handled in chat layer)
    """

    _client: BaseClient | None = PrivateAttr(default=None)

    def get_client(self) -> BaseClient:
        if self._client is None:
            region = os.environ["AWS_REGION"]
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(read_timeout=10000),
            )
        return self._client

    def _get_chat(self, params: LLMParams) -> BaseChatModel:
        model = params.model or os.environ.get("BEDROCK_MODEL_ID", "")
        if not model:
            raise ValueError(
                "Bedrock model not specified. Set params.model or BEDROCK_MODEL_ID env var."
            )
        if "claude" in model.lower():
            return ClaudeChat(
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
                f"Model: {params.model} is not supported by Bedrock LLM Manager"
            )
