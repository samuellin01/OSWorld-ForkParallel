# pyre-strict
from typing import Type

from ..analect import Analect, AnalectRunContext
from . import base as entry_base
from .manager import get_entry
from .mixin import EntryAnalectMixin


class Entry(
    Analect[entry_base.EntryInput, entry_base.EntryOutput], entry_base.EntryBase
):
    async def _run_selected_analect(
        self,
        analect: EntryAnalectMixin,
        inp: entry_base.EntryInput,
        context: AnalectRunContext,
    ) -> entry_base.EntryOutput:
        context.memory_manager.entry_name = (
            analect.display_name() or type(analect).__name__
        )
        out = await context.invoke_analect(analect, inp)
        assert out is not None
        return out

    async def _run_selected_analect_cls(
        self,
        selected_cls: Type[EntryAnalectMixin],
        inp: entry_base.EntryInput,
        context: AnalectRunContext,
    ) -> entry_base.EntryOutput:
        try:
            analect = await selected_cls.new_from_entry_input(inp)
        except Exception as e:
            raise ValueError(
                f"Error creating analect class {selected_cls}: {e}. "
                "Entry analects must be instantiated without any arguments."
            )
        return await self._run_selected_analect(analect, inp, context)

    async def impl(
        self, inp: entry_base.EntryInput, context: AnalectRunContext
    ) -> entry_base.EntryOutput:
        if inp.entry_name is None:
            raise ValueError("entry_name is required")

        entry_cls = get_entry(
            display_name=inp.entry_name, namespace=context.namespace_id
        )
        return await self._run_selected_analect_cls(
            selected_cls=entry_cls,
            inp=inp,
            context=context,
        )
