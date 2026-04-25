# pyre-strict


import bs4
from pydantic import BaseModel, Field

from ....core import types as cf
from ....core.analect import AnalectRunContext

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...tags import Thinking, unescaped_tag_content
from ..tool_use import ToolUseExtension

from .prompt import THINKING_TOOL_USE_PROMPT, THINKING_TOOL_USE_USER_MESSAGE


class ThinkingInput(BaseModel):
    thought: str = Field(..., description="Your thoughts.")


class ThinkingExtension(ToolUseExtension):
    name: str = "think"
    included_in_system_prompt: bool = False
    user_message: str = Field(
        default=THINKING_TOOL_USE_USER_MESSAGE,
        description="User message to send to LLM use when this tool is used",
    )

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        if tag.name == Thinking().name:
            await context.io.divider()
            await context.io.system(
                unescaped_tag_content(tag),
                run_status=cf.RunStatus.COMPLETED,
                run_label="Thinking",
            )
            await context.io.divider()

    @property
    async def tools(self) -> list[ant.ToolLike]:
        if self.enable_tool_use:
            return [
                ant.Tool(
                    name=self.name,
                    description=THINKING_TOOL_USE_PROMPT,
                    input_schema=ThinkingInput.schema(),
                )
            ]

        return []

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        inp = ThinkingInput.parse_obj(tool_use.input)
        await context.io.divider()
        await context.io.system(
            inp.thought,
            run_status=cf.RunStatus.COMPLETED,
            run_label="Thinking",
        )
        await context.io.divider()
        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content="Thoughts recorded",
        )
