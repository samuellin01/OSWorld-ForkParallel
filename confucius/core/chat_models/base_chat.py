# pyre-strict


from typing import Any

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


class RetryableConfig(BaseModel):
    """
    Config for retryable decorator:
    https://fburl.com/code/yfrqxuhv
    """

    retries: int = Field(default=3)
    max_duration: float = Field(default=0.0)
    splay: float = Field(default=1.0)
    sleep_time: float | None = Field(default=None)
    sleep_time_intervals: list[float] | None = Field(default=None)


class ConfuciusBaseChat(BaseLanguageModel):
    model: str = Field(
        ...,
        description="The model name to use for the API",
    )

    temperature: float | None = Field(
        default=None,
        description="Value that controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results.",
    )

    top_p: float | None = Field(
        default=None,
        description="Tokens are selected from the most to least probable until the sum of their probabilities equals this value. Use a lower value for less random responses and a higher value for more random responses.",
    )

    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens that can be generated in the response.",
    )

    stop: list[str] | None = Field(
        None,
        description="List of strings that tells the model to stop generating text if one of the strings is encountered in the response.",
    )

    retryable_config: RetryableConfig = Field(
        default=RetryableConfig(retries=5, sleep_time_intervals=[5, 10, 20, 30, 60]),
        description="The retryable config to use for the API",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return True

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling API."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "stop": self.stop,
        }

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {**self._default_params}
