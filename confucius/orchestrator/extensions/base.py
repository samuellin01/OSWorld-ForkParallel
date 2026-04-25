# pyre-strict


from typing import Any

import bs4
from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel, Field, field_validator

from ...core.analect import AnalectRunContext
from ...core.llm_manager import LLMParams
from ...core.memory import CfMemory, CfMessage
from ..exceptions import OrchestratorInterruption
from ..tags import Example, Examples, Tag, TagLike


class Processor(BaseModel):
    """
    Base Interface to process certain tags
    """

    # pyre-fixme[8]: Incompatible attribute type [8]: Attribute `name` declared in class `Processor` has type `str` but is used as type `None`.
    name: str = Field(default=None, description="Processor name", validate_default=True)
    included_in_system_prompt: bool = Field(
        default=True,
        description="Whether to include the processor description in the system prompt",
    )
    format_instructions: TagLike | None = Field(
        default=None,
        description="The user defined format instructions that will override the automatic generated one",
    )
    examples: list[TagLike] = Field(
        default_factory=list, description="List of examples of the tags"
    )

    # pyre-ignore[56]: Invalid decoration [56]
    @field_validator("name", mode="before")
    @classmethod
    def set_default_name(cls, name: Any) -> str:
        # If 'name' is not provided, set it to the class name
        if name is None:
            return cls.__name__
        return name

    async def description(self) -> TagLike:
        """
        Return the description of the processor.
        """
        tags = []
        if self.format_instructions:
            tags.append(
                Tag(
                    name="format_instructions",
                    contents=self.format_instructions,
                )
            )
        if self.examples:
            tags.append(
                Examples(
                    contents=[Example(contents=example) for example in self.examples]
                )
            )
        return tags


class Extension(BaseModel):
    """
    Interface for orchestrator to extend the functionality of the system.
    """

    # pyre-fixme[8]: Incompatible attribute type [8]: Attribute `name` declared in class `Processor` has type `str` but is used as type `None`.
    name: str = Field(default=None, description="Extension name", validate_default=True)
    included_in_system_prompt: bool = Field(
        default=True,
        description="Whether to include the extension description in the system prompt",
    )

    @property
    def stop_sequences(self) -> list[str]:
        """
        Property to define the stop sequences for the LLM.

        Returns:
            list[str]: A list of stop sequences that the LLM should use to determine when to stop generating text.
        """

        return []

    # pyre-ignore[56]: Invalid decoration [56]
    @field_validator("name", mode="before")
    @classmethod
    def set_default_name(cls, name: Any) -> str:
        # If 'name' is not provided, set it to the class name
        if name is None:
            return cls.__name__
        return name

    async def description(self) -> TagLike:
        """
        Return a extension description, which will be inserted into the orchestrator system prompt.
        """
        return ""

    async def on_plain_text(self, content: str, context: AnalectRunContext) -> None:
        """
        Callback to process the plain texts around tags within the AI response.

        Args:
            content (str): The plain text.

        Returns:
            None
        """
        pass

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        """
        Callback to process the tags.

        Args:
            tag (bs4.Tag): The tag.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            None
        """
        pass

    async def on_input_messages(
        self, messages: list[CfMessage], context: AnalectRunContext
    ) -> list[CfMessage]:
        """
        Callback to process the input messages.

        Args:
            messages (list[CfMessage]): The input messages.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            list[CfMessage]: The processed input messages.
        """
        return messages

    async def on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> list[BaseMessage]:
        """
        Callback to process the LLM invocation.

        Args:
            messages (list[BaseMessage]): The input messages for the LLM.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            list[BaseMessage]: The processed prompt and inputs of the LLM invocation.
        """
        return messages

    async def on_invoke_llm_with_params(
        self,
        messages: list[BaseMessage],
        llm_params: LLMParams,
        context: AnalectRunContext,
    ) -> tuple[list[BaseMessage], LLMParams]:
        """
        Extended callback to process the LLM invocation along with LLM params.
        This provides extensions a chance to customize messages and llm_params.

        Args:
            messages (list[BaseMessage]): The input messages for the LLM.
            llm_params (LLMParams): The parameters for the LLM invocation.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            tuple[list[BaseMessage], LLMParams]: Possibly modified messages and llm_params.
        """
        return messages, llm_params

    async def on_llm_response(
        self,
        message: BaseMessage,
        context: AnalectRunContext,
    ) -> BaseMessage:
        """
        Callback to process the LLM response.
        Different from on_llm_output, this is called before the response is converted to plain text.

        Args:
            message (BaseMessage): The LLM response.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            BaseMessage: The processed LLM response.
        """
        return message

    async def on_llm_output(
        self,
        text: str,
        context: AnalectRunContext,
    ) -> str:
        """
        Callback to process the LLM output.

        Args:
            text (str): The text of the LLM output.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            The processed text of the LLM output.
        """
        return text

    async def on_interruption(
        self, exc: OrchestratorInterruption, context: AnalectRunContext
    ) -> None:
        """
        Callback to process the interruption.

        Args:
            exc (OrchestratorInterruption): The interruption exception.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            None
        """
        pass

    async def on_add_messages(
        self, messages: list[CfMessage], context: AnalectRunContext
    ) -> list[CfMessage]:
        """
        Callback to process the messages to be added to the memory manager.

        Args:
            messages (list[CfMessage]): The messages to be added.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            list[CfMessage]: The processed messages to be added.
        """
        return messages

    async def on_memory(self, memory: CfMemory, context: AnalectRunContext) -> CfMemory:
        """
        Callback to process the memory before converting to LLM prompt.

        Args:
            memory (CfMemory): The memory.
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            The processed memory.
        """
        return memory

    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """
        Callback after processing the messages is complete.
        Raise an OrchestratorInterruption to prompt the orchestrator to continue

        Args:
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            None
        """
        pass

    async def on_session_complete(self, context: AnalectRunContext) -> None:
        """
        Callback after the entire orchestrator session is complete.
        This is called exactly once at the end of impl(), in a finally block,
        regardless of how the session ended (normal completion, termination, or error).

        Args:
            context (AnalectRunContext): The context of the orchestrator.

        Returns:
            None
        """
        pass
