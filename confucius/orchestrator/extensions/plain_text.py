# pyre-strict

from ...core.analect import AnalectRunContext

from .base import Extension


class PlainTextExtension(Extension):
    name: str = "plain_text"
    included_in_system_prompt: bool = False

    async def on_plain_text(self, content: str, context: AnalectRunContext) -> None:
        await context.io.ai(content)
