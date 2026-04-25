# pyre-strict


from typing import Any, Optional

from langchain_core.messages import BaseMessage
from pydantic import Field

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant

from .base import BasePromptCaching


class AnthropicPromptCaching(BasePromptCaching):
    """
    Specialized version for Anthropic APIs
    """

    cache_control: ant.CacheControl = Field(
        default_factory=ant.CacheControl,
        description="The cache control to use for the LLM. Defaults to 5-minute TTL. Set CacheControl(ttl='1h') for 1-hour TTL at 2x base input token price.",
    )

    def is_cache_control_allowed(self, content: dict[str, Any]) -> bool:
        """
        Check if the cache control is allowed for the given content.
        """
        return not (
            content.get("type") == ant.MessageContentType.TEXT
            and content.get("text") == ""
        )

    async def on_adding_cache_breakpoint(
        self, message: BaseMessage
    ) -> Optional[BaseMessage]:
        """
        Called when a cache break point is added to the message.
        Returns the updated message with the cache control added if successful, otherwise None.
        """
        contents = []
        msg_content = message.content
        assert isinstance(msg_content, (list, str))
        if isinstance(msg_content, list):
            for ct in msg_content:
                assert isinstance(ct, (dict, str))
                if isinstance(ct, dict):
                    contents.append(ct)
                else:
                    contents.append(ant.MessageContentText(text=ct).dict())
        else:
            contents.append(ant.MessageContentText(text=msg_content).dict())

        cache_added = False
        for content in reversed(contents):
            if self.is_cache_control_allowed(content):
                content["cache_control"] = self.cache_control.dict()
                cache_added = True
                break

        return message.copy(update={"content": contents}) if cache_added else None
