# pyre-strict

from __future__ import annotations

from enum import Enum
import logging
from typing import Any

import aioconsole

from pydantic import BaseModel, ConfigDict, Field
from rich.align import AlignMethod

from rich.console import Console, JustifyMethod, OverflowMethod
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from .. import types as cf

from .base import IOInterface

logger: logging.Logger = logging.getLogger(__name__)


class IOState(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


class StdIOInterface(IOInterface, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """Console-based IO implementation using built-in print/input."""

    state: IOState | None = Field(None, description="Current state of IO")
    console: Console = Field(
        default_factory=lambda: Console(),
        description="The console for printing",
    )

    async def rule(
        self,
        text: str,
        *,
        align: AlignMethod = "left",
        style: str | Style = "rule.line",
        characters: str = "─",
        **kwargs: Any,
    ) -> None:
        self.console.rule(text, align=align, style=style, characters=characters)

    async def divider(self) -> None:
        await self.rule("")

    async def print(
        self,
        text: str,
        *,
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self.console.print(
                text,
                style=style,
                justify=justify,
                overflow=overflow,
            )
        except Exception as e:
            logger.error(f"Error printing: {e}\n--\nText: {text}\nStyle: {style}")

    async def human(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        if self.state != IOState.HUMAN:
            await self.rule("HUMAN")
        self.state = IOState.HUMAN
        await super().human(text, **kwargs)

    async def ai(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        if self.state != IOState.AI:
            await self.rule("AI")
        self.state = IOState.AI
        await super().ai(text, **kwargs)

    async def system(
        self,
        text: str,
        *,
        progress: int | None = None,
        run_status: cf.RunStatus | None = None,
        run_label: str | None = None,
        run_description: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self.state != IOState.SYSTEM:
            await self.rule("SYSTEM")
        self.state = IOState.SYSTEM
        if "style" not in kwargs:
            kwargs["style"] = "bright_black"
        await super().system(text, **kwargs)

    async def vizir(
        self,
        query_map: str,
        view: str,
        *,
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        **kwargs: Any,
    ) -> None:
        self.console.print(
            f"Query Map: {query_map}\nView: {view}",
            style=style,
            justify=justify,
            overflow=overflow,
        )

    async def error(
        self,
        text: str,
        *,
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        **kwargs: Any,
    ) -> None:
        self.console.print(
            Panel(
                self.console.highlighter(Text(text, **kwargs)),
                title="Error",
                title_align="left",
                border_style="red",
            ),
            style=style,
            justify=justify,
            overflow=overflow,
        )
        self.state = None

    async def log(
        self,
        text: str,
        *,
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        **kwargs: Any,
    ) -> None:
        self.console.print(
            text,
            highlight=False,
            style=style,
            justify=justify,
            overflow=overflow,
        )

    async def _get_input(self, prompt: str, placeholder: str | None = None) -> str:
        placeholder = f"[{placeholder}]" if placeholder else ""
        return await aioconsole.ainput(f"{prompt} {placeholder} > ")

    async def on_cancel(self) -> None:
        await self.system("Interrupted...")
