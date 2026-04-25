# pyre-strict


from unittest.mock import Mock

from botocore.client import BaseClient
from pydantic import Field

from ..base_chat import ConfuciusBaseChat


class BedrockBase(ConfuciusBaseChat):
    client: BaseClient | Mock = Field(
        ..., description="The AWS client to use for the Bedrock API"
    )

    version: str = Field(
        "bedrock-2023-05-31", description="The version of the API to use"
    )

    beta: list[str] | None = Field(
        None,
        description="Beta features to use when generating the output",
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "bedrock"
