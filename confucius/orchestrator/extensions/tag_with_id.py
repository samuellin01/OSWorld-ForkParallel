# pyre-strict

import html
import re

import bs4
from pydantic import Field, PrivateAttr

from ...core.analect import AnalectRunContext
from ...core.memory import CfMessage
from ..exceptions import OrchestratorInterruption
from ..tags import unescape, unescaped_tag_content
from .base import Extension


class TagWithIDExtension(Extension):
    tag_name: str = Field(..., description="The tag name that the extension handles")
    default_identifier: str | None = Field(None, description="The default identifier")
    escape_tag_content: bool = Field(
        True, description="Whether to escape the content of the tag"
    )
    _tag_pattern: re.Pattern[str] | None = PrivateAttr(None)

    @property
    def stop_sequences(self) -> list[str]:
        return [f"</{self.tag_name}>"]

    @property
    def tag_pattern(self) -> re.Pattern[str]:
        if self._tag_pattern is None:
            self._tag_pattern = re.compile(
                rf"(?P<opening_tag><{self.tag_name}(?:\s+[^>]*)?>)"
                rf"(?P<content>.*?)"
                rf"(?P<closing_tag></{self.tag_name}>)",
                flags=re.DOTALL,
            )
        return self._tag_pattern

    async def on_add_messages(
        self, messages: list[CfMessage], context: AnalectRunContext
    ) -> list[CfMessage]:
        for msg in messages:
            if isinstance(msg.content, str):
                msg.content = unescape(msg.content)
        return messages

    async def on_llm_output(
        self,
        text: str,
        context: AnalectRunContext,
    ) -> str:
        if self.escape_tag_content:

            def escape_match(match: re.Match[str]) -> str:
                opening_tag: str = match.group("opening_tag")
                content: str = match.group("content")
                closing_tag: str = match.group("closing_tag")

                escaped_content: str = html.escape(content, quote=False)
                return f"{opening_tag}{escaped_content}{closing_tag}"

            return self.tag_pattern.sub(escape_match, text)
        return text

    async def on_tag_with_id(
        self,
        identifier: str,
        content: str,
        context: AnalectRunContext,
        attrs: dict[str, str] | None = None,
    ) -> None:
        """
        Callback to process the tag with id.

        Args:
            identifier (str): The identifier of the tag.
            content (str): The content of the tag.
            context (AnalectRunContext): The context of the collector.
            attrs (dict[str, str]): The attributes of the tag.

        Returns:
            None
        """
        pass

    async def is_valid_tag(self, tag: bs4.Tag) -> bool:
        return tag.name == self.tag_name

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        if not (await self.is_valid_tag(tag)):
            return

        try:
            identifier: str = (
                tag.get("identifier", self.default_identifier)
                if self.default_identifier
                else tag["identifier"]
            )
            content: str = unescaped_tag_content(tag)
            await self.on_tag_with_id(
                identifier=identifier,
                content=content,
                context=context,
                attrs=tag.attrs,
            )

        except KeyError as exc:
            raise OrchestratorInterruption(
                messages=[
                    CfMessage(content=f"Invalid tag, missing required attribute: {exc}")
                ]
            )
