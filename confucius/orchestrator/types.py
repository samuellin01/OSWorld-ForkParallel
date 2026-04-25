# pyre-strict

from pydantic import BaseModel, Field

from ..core.memory import CfMessage


class OrchestratorInput(BaseModel):
    messages: list[CfMessage] = Field(
        [], description="additional input messages, for more flexible inputs"
    )
    task: str = Field("", description="task of the orchestrator")

    class Config:
        arbitrary_types_allowed = True


class OrchestratorOutput(BaseModel):
    """
    Orchestrator doesn't have any output since it will return at each round of interaction.

    You will need to use session storage, memory or artifacts to store the output of the orchestrator.
    """

    pass
