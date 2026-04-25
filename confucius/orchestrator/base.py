# pyre-strict


from abc import ABC, abstractmethod

import bs4
from pydantic import Field, PrivateAttr

from ..core import types as cf
from ..core.analect.analect import Analect, AnalectRunContext
from ..core.memory import CfMessage

from .exceptions import (
    MaxIterationsReachedError,
    OrchestratorInterruption,
    OrchestratorTermination,
)
from .extensions import Extension
from .types import OrchestratorInput, OrchestratorOutput


class BaseOrchestrator(Analect[OrchestratorInput, OrchestratorOutput], ABC):
    extensions: list[Extension] = Field(
        default_factory=list,
        description="Extensions to extend the functionality of the system",
    )
    max_iterations: int | None = Field(
        default=250,
        description="Maximum number of iterations to call the LLM. If None, no limit is set.",
    )

    _num_iterations: int = PrivateAttr(default=0)

    class Config:
        arbitrary_types_allowed = True

    @property
    def stop_sequences(self) -> list[str]:
        """
        Property to define the stop sequences for the LLM. Default is the union of all stop sequences from the extensions (deduplicated).

        Returns:
            list[str]: A list of stop sequences that the LLM should use to determine when to stop generating text.
        """
        return list({stop for ext in self.extensions for stop in ext.stop_sequences})

    async def on_plain_text(self, content: str, context: AnalectRunContext) -> None:
        """
        Callback to process the plain texts around tags within the AI response.
        Default behavior is sequential processing of the plain text based on the order of the extensions.

        Args:
            content (str): The plain text.

        Returns:
            None
        """
        for ext in self.extensions:
            await ext.on_plain_text(content, context)

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        """
        Callback to process the tags. Default behavior is sequential processing of the tags based on the order of the extensions.

        Args:
            tag (bs4.Tag): The tag.
            context (AnalectRunContext): The context of the collector.

        Returns:
            None
        """
        for ext in self.extensions:
            await ext.on_tag(tag, context)

    async def on_input_messages(
        self, messages: list[CfMessage], context: AnalectRunContext
    ) -> list[CfMessage]:
        """
        Callback to process the input messages. Default behavior is sequential processing of the messages based on the order of the extensions.

        Args:
            messages (list[CfMessage]): The input messages.
            context (AnalectRunContext): The context of the collector.

        Returns:
            list[CfMessage]: The processed input messages.
        """
        for ext in self.extensions:
            messages = await ext.on_input_messages(messages, context)
        return messages

    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """
        Callback after processing the messages is complete. Default behavior is sequentially call each extension's on_process_messages_complete.

        Args:
            context (AnalectRunContext): The context of the collector.

        Raise an OrchestratorInterruption to prompt the orchestrator to continue
        """
        for ext in self.extensions:
            await ext.on_process_messages_complete(context)

    async def on_session_complete(self, context: AnalectRunContext) -> None:
        """
        Callback after the entire orchestrator session is complete.
        Default behavior is sequentially call each extension's on_session_complete.

        Args:
            context (AnalectRunContext): The context of the orchestrator.
        """
        for ext in self.extensions:
            await ext.on_session_complete(context)

    async def on_interruption(
        self, exc: OrchestratorInterruption, context: AnalectRunContext
    ) -> None:
        """
        Callback to process the interruption. Default behavior is sequential processing of the interruption based on the order of the extensions.

        Args:
            exc (OrchestratorInterruption): The interruption exception.
            context (AnalectRunContext): The context of the collector.

        Returns:
            None
        """
        for ext in self.extensions:
            await ext.on_interruption(exc, context)

    async def add_messages(
        self,
        messages: list[CfMessage],
        context: AnalectRunContext,
    ) -> None:
        """
        Add messages to the memory manager. Default behavior is sequential processing of the messages based on the order of the extensions.

        Args:
            messages (list[CfMessage]): The messages to add.
            context (AnalectRunContext): The context of the collector.

        Returns:
            None
        """
        for ext in self.extensions:
            messages = await ext.on_add_messages(messages, context)

        if messages:
            context.memory_manager.add_messages(messages)

    @abstractmethod
    async def get_root_tag(self, task: str, context: AnalectRunContext) -> bs4.Tag:
        """
        Abstract method to retrieve the root tag for a given task.

        This method should be implemented by subclasses to define how the root tag
        is obtained based on the task and context provided. The root tag is used
        for further processing within the orchestrator.

        Args:
            task (str): The task for which the root tag is to be retrieved.
            context (AnalectRunContext): The context in which the task is being executed.

        Returns:
            bs4.Tag: The root tag corresponding to the task.
        """

        pass

    async def _process_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        await self.add_messages(
            [
                CfMessage(
                    type=cf.MessageType.AI,
                    content=tag.prettify(
                        formatter=bs4.formatter.HTMLFormatter(indent=0)
                    ),
                    attachments=[],
                )
            ],
            context,
        )

        await self.on_tag(tag, context)

    async def _process_plain_text(self, text: str, context: AnalectRunContext) -> None:
        await self.add_messages(
            [
                CfMessage(
                    type=cf.MessageType.AI,
                    content=text,
                    attachments=[],
                )
            ],
            context,
        )
        await self.on_plain_text(text, context)

    async def _process_root_tag(
        self,
        tag: bs4.Tag,
        context: AnalectRunContext,
    ) -> None:
        for child in tag.children:
            if isinstance(child, bs4.Tag):
                await self._process_tag(child, context)
            elif isinstance(child, bs4.NavigableString):
                text = child.strip()
                if text:
                    await self._process_plain_text(text, context)
            else:
                raise ValueError(f"Unknown child type: {type(child)}")

    async def _process_interruption(
        self, exc: OrchestratorInterruption, context: AnalectRunContext
    ) -> None:
        # pyre-fixme[6]: Incompatible parameter type [6]: In call `BaseOrchestrator.add_messages`, for 1st positional argument, expected `List[CfMessage]` but got `Union[List[CfMessage], List[Variable[_T]]]`.
        await self.add_messages(exc.messages or [], context)
        await self.on_interruption(exc, context)

    async def _process_messages(self, task: str, context: AnalectRunContext) -> None:
        if (
            self.max_iterations is not None
            and self._num_iterations >= self.max_iterations
        ):
            raise MaxIterationsReachedError(
                f"Maximum number of iterations reached: {self.max_iterations}, Please start a new session with updated context."
            )
        root_tag = await self.get_root_tag(task, context)
        self._num_iterations += 1

        try:
            await self._process_root_tag(root_tag, context)
            await self.on_process_messages_complete(context)
        except OrchestratorInterruption as exc:
            await self._process_interruption(exc, context)
            # Recursively process the messages until we reach a non-interruption
            await self._process_messages(task, context)

    async def impl(
        self, inp: OrchestratorInput, context: AnalectRunContext
    ) -> OrchestratorOutput:
        try:
            inp_msgs = await self.on_input_messages(inp.messages, context)
            await self.add_messages(inp_msgs, context)
            await self._process_messages(inp.task, context)
        except OrchestratorTermination:
            pass
        finally:
            await self.on_session_complete(context)

        return OrchestratorOutput()
