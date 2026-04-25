# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import cast, List

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from ....core.analect import AnalectRunContext, get_current_context

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..tool_use import ToolUseExtension


LLM_CALL_COUNTER_KEY = "llm_call_counter"

DEFAULT_REMINDER_MESSAGE = """\
<reminder>
Please consider writing or editing memory to keep a persistent record of your work.
</reminder>
"""

SNOOZE_NOTE_MESSAGE = """\

<note>
You can use the "snooze_reminder" tool to snooze the reminder temporarily.
</note>
"""


class _LLMCallCounter(BaseModel):
    num_llm_calls: int = Field(
        default=0,
        description="Number of LLM calls since the beginning of the session or since the last reset",
    )
    total_num_llm_calls: int = Field(
        default=0,
        description="Total number of LLM calls since the beginning of the session",
    )


class MemoryReminder(ToolUseExtension):
    max_llm_calls_before_reminder: int = Field(
        default=25,
        description="Maximum number of LLM calls before reminder of writing or editing memory",
    )
    reminder_message: str = Field(
        default=DEFAULT_REMINDER_MESSAGE,
        description="Message to remind the user to write or edit memory",
    )
    snooze_amount: int = Field(
        default=10,
        description="Number of LLM calls to reduce when snoozing the reminder",
    )

    @model_validator(mode="after")
    def validate_snooze_amount(self) -> "MemoryReminder":  # noqa
        """Validate that snooze_amount is not greater than max_llm_calls_before_reminder."""
        if self.snooze_amount > self.max_llm_calls_before_reminder:
            raise ValueError(
                f"snooze_amount ({self.snooze_amount}) cannot be greater than "
                f"max_llm_calls_before_reminder ({self.max_llm_calls_before_reminder})"
            )
        return self

    @property
    def _llm_call_counter(self) -> _LLMCallCounter:
        """
        Counters for LLM calls in the current session
        """
        context = get_current_context()
        return cast(
            _LLMCallCounter,
            context.session_storage[self.__class__.__name__].setdefault(
                LLM_CALL_COUNTER_KEY, _LLMCallCounter()
            ),
        )

    async def on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> list[BaseMessage]:
        self._llm_call_counter.num_llm_calls += 1
        self._llm_call_counter.total_num_llm_calls += 1
        messages = await super().on_invoke_llm(messages, context)

        if self._llm_call_counter.num_llm_calls >= self.max_llm_calls_before_reminder:
            reminder_message = self.reminder_message + SNOOZE_NOTE_MESSAGE

            messages.append(HumanMessage(content=reminder_message))

        return messages

    def reset_reminder(self) -> None:
        self._llm_call_counter.num_llm_calls = 0

    @property
    async def tools(self) -> List[ant.ToolLike]:
        """Return the snooze_reminder tool if tool use is enabled."""
        if self.enable_tool_use:
            return [
                ant.Tool(
                    name="snooze_reminder",
                    description="Snooze the memory reminder temporarily. This tool requires no arguments.",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]
        return []

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Handle the snooze_reminder tool."""
        if tool_use.name == "snooze_reminder":
            # Decrease num_llm_calls by snooze_amount, but don't go below 0
            current_count = self._llm_call_counter.num_llm_calls
            new_count = max(0, current_count - self.snooze_amount)
            self._llm_call_counter.num_llm_calls = new_count

            await context.io.ai("", warning_message="Memory reminder snoozed")

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content="Memory reminder has been snoozed successfully.",
            )

        # This should not happen if tools() is correctly implemented
        raise ValueError(f"Unknown tool: {tool_use.name}")
