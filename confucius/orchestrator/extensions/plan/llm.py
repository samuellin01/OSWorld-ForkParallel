# pyre-strict

import html
from enum import Enum

import bs4
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, model_validator, PrivateAttr

from ....core import types as cf

from ....core.analect import AnalectRunContext
from ....core.llm_manager import LLMParams
from ....core.memory import CfMemory, CfMessage
from ..token.estimator import TokenEstimatorExtension

from .prompts import LLM_CODING_ARCHITECT_PROMPT
from .utils import prompt_to_convo_tag


class MessageFilterStatus(str, Enum):
    WILL_BE_OMITTED = "will_be_omitted"
    OMITTED = "omitted"


__FILTER_STATUS_KEY__ = "filter_status"


class LLMPlannerExtension(TokenEstimatorExtension):
    name: str = "llm_planner"
    included_in_system_prompt: bool = False
    llm_params: LLMParams = Field(
        default=LLMParams(
            model="claude-sonnet-4",
            temperature=0.3,
            top_p=0.7,
            initial_max_tokens=4096,
        ),
        description="The LLM parameters to use for the planner.",
    )
    prompt: ChatPromptTemplate = Field(
        ...,
        description="The messages to be sent to the LLM. This can be a list of strings or a list of CfMessage objects.",
    )
    plan_tag_name: str = Field(
        default="plan", description="The tag name to use for the plan."
    )
    step_tag_name: str = Field(
        default="next_step", description="The tag name to use for the plan step."
    )
    summary_tag_name: str = Field(
        default="summary", description="The tag name to use for the plan summary."
    )
    plan_suffix: str = Field(
        default="Now, based on the plan, I will",
        description="The suffix to add to the plan to trigger additional LLM outputs. Otherwise, the LLM may stop after the plan.",
    )
    additional_messages: list[CfMessage] = Field(
        default_factory=list,
        description="Additional messages to be appended",
    )
    # As a result of max_prompt_length and min_prompt_length, the processed prompt length will be generally in the range of [min_prompt_length, max_prompt_length]
    max_prompt_length: int = Field(
        default=180000,
        description="The threshold of maximum length of the prompt, if exceeded, the planner will step-in and summarize the plan, otherwise, it will skip, (unit: tokens)",
    )
    min_prompt_length: int = Field(
        default=50000,
        description="The threshold of minimum length of the prompt, when planner is triggered, the prompt will be trimmed to less or equal to this length (unit: tokens)",
    )
    max_num_messages: int = Field(
        default=500,
        description="The maximum number of messages to be included in the prompt, if exceeded, the planner will step-in and summarize the plan",
    )
    start_index: int = Field(
        default=1,
        description="The index of the first message to be omitted, default to 1 to skip the first message which is the initial user request",
    )
    # If this list is empty, the planner will not be triggered
    _messages_to_be_omitted: list[CfMessage] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def validate_prompt_length(self) -> "LLMPlannerExtension":  # noqa: B902
        if self.max_prompt_length < self.min_prompt_length:
            raise ValueError(
                f"max_prompt_length {self.max_prompt_length} must be greater than or equal to min_prompt_length {self.min_prompt_length}"
            )
        return self

    class Config:
        arbitrary_types_allowed = True

    @property
    def stop_sequences(self) -> list[str]:
        return [f"</{self.plan_tag_name}>"]

    async def on_memory(self, memory: CfMemory, context: AnalectRunContext) -> CfMemory:
        memory.messages = [
            msg
            for msg in memory.messages
            if msg.additional_kwargs.get(__FILTER_STATUS_KEY__)
            != MessageFilterStatus.OMITTED.value
        ]
        prompt_lengths = await self.get_prompt_token_lengths(memory.messages)
        total_length = sum(prompt_lengths)
        if (
            total_length <= self.max_prompt_length
            and len(memory.messages) <= self.max_num_messages
        ):
            # No need to plan
            self._messages_to_be_omitted.clear()
            return memory

        for i in range(self.start_index, len(memory.messages)):
            if (
                total_length <= self.min_prompt_length
                and memory.messages[i].type == cf.MessageType.AI
            ):
                break
            memory.messages[i].additional_kwargs[
                __FILTER_STATUS_KEY__
            ] = MessageFilterStatus.WILL_BE_OMITTED.value
            self._messages_to_be_omitted.append(memory.messages[i])
            total_length -= prompt_lengths[i]
        return memory

    @property
    def _plan_inputs(self) -> dict[str, str]:
        return {
            "plan_tag_name": self.plan_tag_name,
            "step_tag_name": self.step_tag_name,
            "summary_tag_name": self.summary_tag_name,
        }

    def _get_plan_prompt(self, messages: list[BaseMessage]) -> ChatPromptTemplate:
        return self.prompt + prompt_to_convo_tag(messages)

    async def _get_plan_text(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> str:
        plan_text: str | None = None
        plan_prompt = self._get_plan_prompt(messages)
        chat = context.llm_manager._get_chat(params=self.llm_params)
        while True:
            messages = plan_prompt.format_messages(**self._plan_inputs)
            response = await context.invoke(chat, messages)
            content = response.content
            res = (
                content
                if isinstance(content, str)
                else "\n".join(
                    ct.get("text", "") if isinstance(ct, dict) else ct for ct in content
                )
            )
            soup = bs4.BeautifulSoup(res, "html.parser")
            plan_tag = soup.find(name=self.plan_tag_name)
            summary_tag = soup.find(name=self.summary_tag_name)
            if plan_tag is None or summary_tag is None:
                err_msg = f"No <{self.plan_tag_name}> or <{self.summary_tag_name}> tag found. Please write your plan in the <{self.plan_tag_name}> tag and summary in the <{self.summary_tag_name}> tag."
                await context.io.system("---\n" + err_msg, run_label="Planning...")
                plan_prompt += [
                    AIMessage(content=res),
                    HumanMessage(content=err_msg),
                ]
            else:
                plan_text = html.unescape(summary_tag.prettify())
                first_step_tag = plan_tag.find(self.step_tag_name, recursive=False)
                if first_step_tag:
                    # Get the next sibling after the first step
                    current = first_step_tag.next_sibling
                    # Remove all following siblings
                    while current:
                        next_sibling = current.next_sibling
                        current.extract()
                        current = next_sibling

                    plan_text += html.unescape(plan_tag.prettify())

                break
        assert plan_text is not None
        return plan_text

    async def _on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
        planning_reason: str | None = None,
    ) -> list[BaseMessage]:
        if not self._messages_to_be_omitted:
            return messages

        await context.io.divider()
        planning_reason = (
            planning_reason
            or f"Generating Plan due to current prompt length exceeding the token limit: {self.max_prompt_length}"
        )
        await context.io.system(
            planning_reason,
            run_label="Planning...",
        )

        try:
            plan_text = await self._get_plan_text(messages, context)
        except Exception as exc:
            await context.io.system(
                f"Failed to generate plan due to {type(exc).__name__}: {str(exc)}",
                run_label="Planning failed",
                run_status=cf.RunStatus.FAILED,
            )
            raise exc

        await context.io.system(
            f"---\nPlan Generated:\n{plan_text}",
            run_status=cf.RunStatus.COMPLETED,
            run_label="Plan Generated",
        )
        # Adding the suffix to trigger additional LLM outputs
        plan_text += self.plan_suffix
        # The plan summary should be added as a HUMAN message to provide context for the main LLM
        # rather than as an AI message which would require a thinking block when thinking is enabled
        plan_msg = CfMessage(content=plan_text, type=cf.MessageType.HUMAN)
        new_messages = [plan_msg] + [
            # Here we reconstruct the message to make sure the sequence id is incremental
            CfMessage.parse_obj(msg.dict())
            for msg in self.additional_messages
        ]
        context.memory_manager.add_messages(new_messages)

        messages = [
            msg
            for msg in messages
            if (
                msg.additional_kwargs.get(__FILTER_STATUS_KEY__)
                != MessageFilterStatus.WILL_BE_OMITTED.value
            )
        ] + [lc_msg for msg in new_messages for lc_msg in (await msg.to_lc_messages())]

        for msg in self._messages_to_be_omitted:
            msg.additional_kwargs[__FILTER_STATUS_KEY__] = (
                MessageFilterStatus.OMITTED.value
            )
        self._messages_to_be_omitted.clear()

        return messages


class LLMCodingArchitectExtension(LLMPlannerExtension):
    prompt: ChatPromptTemplate = Field(
        default=LLM_CODING_ARCHITECT_PROMPT,
        description="The messages to be sent to the LLM. This can be a list of strings or a list of CfMessage objects.",
    )
