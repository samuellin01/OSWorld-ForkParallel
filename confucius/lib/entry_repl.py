# pyre-strict
import asyncio
import sys
from traceback import extract_tb

from typing import Any

from ..core.entry.base import EntryInput

from ..core.entry.entry import Entry
from .confucius import Confucius


async def run_entry_repl(cf: Confucius, entry_name: str, **kwargs: Any) -> None:
    while True:
        entry_text = await cf.io.get_input(prompt="", placeholder="Send a message ...")
        try:
            await cf.invoke_analect(
                Entry(**kwargs), EntryInput(question=entry_text, entry_name=entry_name)
            )
        except asyncio.CancelledError:
            await cf.io.on_cancel()
            if cf.exiting:
                raise
        except Exception as exc:
            tb = exc.__traceback__
            tb_str = "\n".join(extract_tb(tb).format())
            print(tb_str, file=sys.stderr)
            await cf.io.error(f"{exc}. Check stderr for details")
        finally:
            await cf.save(raise_exception=False)
