# pyre-strict
from typing import Any

from pydantic import BaseModel, Field

from .constants import AWS_CLOUD_ACCOUNT_ID, GOOGLE_CLOUD_PROJECT_ID


class LLMParams(BaseModel):
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    temperature: float = Field(default=0, ge=0, le=1)

    # By default, Confucius will start from a small max token value
    # and perform perform exponential backoffs.
    #
    # User can override the starting value with `initial_max_tokens`,
    # and the maximum value with `max_tokens`. This is useful when
    # you already know the output is going to exceed the starting max_token
    # value.
    initial_max_tokens: int | None = Field(default=None)
    max_tokens: int | None = Field(default=None)

    verbose: bool = Field(default=False)
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    top_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    model: str | None = Field(None, description="Model name")
    repetition_penalty: float = Field(
        default=1.0,
        # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
        ge=0.0,
        # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
        le=2.0,
        description="Repetition penalty for the generation.",
    )
    guided_decode_json_schema: str | None = Field(
        None, description="Guided decode json schema"
    )
    strict: bool = Field(
        default=False,
        description="Strict mode where json schema will be enforced whenever possible.",
    )
    stop: list[str] | None = Field(
        None,
        description="Stop sequences to use when generating the output",
    )

    cache: bool = Field(
        default=False,
        description=(
            "If set to true, we will prefer to use the cached result. "
            "If set to false (default), we will perform fresh LLMs even if the input has been seen before."
        ),
    )

    aws_account_id: str = Field(
        default=AWS_CLOUD_ACCOUNT_ID, description="AWS account id"
    )
    google_cloud_project_id: str = Field(
        default=GOOGLE_CLOUD_PROJECT_ID, description="Google Cloud project id"
    )
    azure_openai_secret_group: str = Field(
        default="", description="Azure OpenAI secret group"
    )

    additional_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments to pass to the LLM call. "
            "This is useful when you want to pass in custom parameters "
            "that are not supported by the LLMParams class."
        ),
    )

    class Config:
        arbitrary_types_allowed = True
