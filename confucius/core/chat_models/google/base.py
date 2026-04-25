# pyre-strict


from unittest.mock import Mock

from google import genai
from pydantic import Field

from ..base_chat import ConfuciusBaseChat


class GoogleBase(ConfuciusBaseChat):
    client: genai.Client | Mock = Field(
        ..., description="The AWS client to use for the Bedrock API"
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
        return "google"
