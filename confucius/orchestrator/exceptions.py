# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Any

from ..core.memory import CfMessage


class OrchestratorInterruption(Exception):
    """
    Interrupt the orchestrator with some messages, all LLM output after the interruption will be ignored
    """

    def __init__(
        self,
        messages: list[CfMessage] | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(messages, str):
            messages = [CfMessage(content=messages)]

        self.messages: list[CfMessage] = messages or []
        super().__init__(*args, **kwargs)


class OrchestratorTermination(Exception):
    pass


class MaxIterationsReachedError(Exception):
    pass
