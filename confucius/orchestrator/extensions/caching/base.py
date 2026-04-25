# pyre-strict

from abc import ABC, abstractmethod
from typing import cast, Optional, Union

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import Field

from ....core.analect import AnalectRunContext, get_current_context

from ....core.memory import CfMemory, CfMessage
from ..token.estimator import TokenEstimatorExtension


CACHE_BREAKPOINT_KEY = "__cache_breakpoint__"
LAST_CHECKPOINT_KEY = "_last_checkpoint__"


class BasePromptCaching(TokenEstimatorExtension, ABC):
    """
    Optimally adding prompt caching break points to the memory and messages for better performance and less token costs.

    Check https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html for more details.
    """

    included_in_system_prompt: bool = False
    min_prompt_length: int = Field(
        default=10000,
        description="The minimum length of the prompt to create a cache break point, (unit: tokens).",
    )
    max_num_checkpoints: int = Field(
        default=4,
        description="The maximum number of cache break points to create.",
        # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
        gt=0,
    )
    cache_system_prompt: bool = Field(
        default=True,
        description="Whether to cache the system prompt.",
    )

    def get_last_checkpoint(self) -> int:
        """
        Use a last_checkpoint value for the entire session
        """
        context = get_current_context()
        return cast(
            int,
            context.session_storage[self.__class__.__name__].setdefault(
                LAST_CHECKPOINT_KEY, 0
            ),
        )

    def set_last_checkpoint(self, value: int) -> None:
        context = get_current_context()
        context.session_storage[self.__class__.__name__][LAST_CHECKPOINT_KEY] = value

    async def on_memory(self, memory: CfMemory, context: AnalectRunContext) -> CfMemory:
        """
        Add cache breakpoints to memory when sufficient new content has been added.
        """
        last_prompt_token_length = self.get_last_prompt_token_length()
        if last_prompt_token_length is not None:
            total_length = last_prompt_token_length
        else:
            total_length = sum(await self.get_prompt_token_lengths(memory.messages))
        
        # Only consider adding a breakpoint if we've accumulated enough new tokens
        if total_length > self.get_last_checkpoint() + self.min_prompt_length:
            if len(memory.messages) > 0:
                # Add breakpoint to the last message
                memory.messages[-1].additional_kwargs[CACHE_BREAKPOINT_KEY] = True

                # Update checkpoint tracking
                self.set_last_checkpoint(total_length)

        return memory

    @abstractmethod
    async def on_adding_cache_breakpoint(
        self, message: BaseMessage
    ) -> Optional[BaseMessage]:
        """
        Called when a cache break point is added to the message.
        """
        ...

    async def _on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> list[BaseMessage]:
        num_checkpoints_to_add = self.max_num_checkpoints

        if self.cache_system_prompt:
            if len(messages) > 0 and isinstance(messages[0], SystemMessage):
                updated_message = await self.on_adding_cache_breakpoint(messages[0])
                if updated_message is not None:
                    messages[0] = updated_message
                    num_checkpoints_to_add -= 1

        base_messages = [
            (i, msg) for i, msg in enumerate(messages) if isinstance(msg, BaseMessage)
        ]
        while base_messages and num_checkpoints_to_add > 0:
            i, msg = base_messages[-1]
            if (
                CACHE_BREAKPOINT_KEY in msg.additional_kwargs
                and msg.additional_kwargs[CACHE_BREAKPOINT_KEY]
            ):
                # add a single checkpoint, iterate messages backwards until we add one
                while base_messages:
                    i, msg = base_messages.pop()
                    updated_message = await self.on_adding_cache_breakpoint(msg)
                    if updated_message is not None:
                        messages[i] = updated_message
                        num_checkpoints_to_add -= 1
                        break
            else:
                base_messages.pop()

        return messages
