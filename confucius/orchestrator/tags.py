# pyre-strict
from __future__ import annotations

import html

from textwrap import dedent

from typing import Any, Dict, List

import bs4
from pydantic import BaseModel, Field


class Tag(BaseModel):
    name: str = Field(..., description="Tag name")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Tag attributes"
    )
    contents: str | Tag | List[str | Tag] | None = Field(
        default_factory=list, description="Tag contents"
    )

    def to_bs4(self, soup: bs4.BeautifulSoup) -> bs4.Tag:
        """
        Recursively converts the Tag object to a BeautifulSoup Tag.
        """
        new_tag = soup.new_tag(self.name, attrs=self.attributes)

        if self.contents:
            if isinstance(self.contents, (str, Tag)):
                self.contents = [self.contents]

            for content in self.contents:
                if isinstance(content, Tag):
                    new_tag.append(content.to_bs4(soup))
                elif isinstance(content, str):
                    new_tag.append(bs4.NavigableString(content))
                else:
                    raise TypeError(f"Unsupported content type: {type(content)}")

        return new_tag

    def prettify(
        self, parser: str = "html.parser", unescape: bool = True, **kwargs: Any
    ) -> str:
        """
        Converts the Tag object to a prettified XML string.

        Args:
            parser (str): The parser to use for the BeautifulSoup object.
            unescape (bool): Whether to unescape the XML string. Defaults to True.
            kwargs: Additional arguments to pass to the BeautifulSoup prettify method.

        Returns:
            str: The prettified XML string.
        """
        soup = bs4.BeautifulSoup("", parser)
        soup.append(self.to_bs4(soup))
        if "formatter" not in kwargs:
            kwargs["formatter"] = bs4.formatter.HTMLFormatter(indent=0)
        result = soup.prettify(**kwargs)
        return html.unescape(result) if unescape else result


TagLike = Tag | str | List[Tag | str]

# Util functions


def unescape(content: str) -> str:
    """
    Unescape HTML special characters in the content.
    It may unescape multiple times until it converges.

    Args:
        content (str): The content to be processed

    Returns:
        str: The content with HTML special characters unescaped
    """
    while content != html.unescape(content):
        content = html.unescape(content)
    return content


def unescaped_tag_content(tag: bs4.element.Tag) -> str:
    """
    Extracts and unescapes the content of a BeautifulSoup Tag.

    Args:
        tag (bs4.element.Tag): The BeautifulSoup Tag from which to extract content.

    Returns:
        str: The unescaped and dedented content of the Tag.
    """

    return dedent(unescape(tag.decode_contents())).strip()


# Some commly used tags for prompts


class Example(Tag):
    name: str = "example"


class Examples(Tag):
    name: str = "examples"


class Thinking(Tag):
    name: str = "thinking"


class Quote(Tag):
    name: str = "quote"


class Reflection(Tag):
    name: str = "reflection"


class AssistantResponse(Tag):
    name: str = "assistant_response"


class UserQuery(Tag):
    name: str = "user_query"


class ToolUse(Tag):
    name: str = "tool_use"


class RedactedThinking(Tag):
    name: str = "redacted_thinking"
