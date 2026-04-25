# pyre-strict
from typing import List, Sequence, Tuple

from ..analect import Analect
from .base import EntryAnalectBase, EntryInput, EntryOutput


class EntryAnalectMixin(EntryAnalectBase):
    """
    See some developer discussions about EntryMixin design here -
    https://docs.google.com/document/d/1Kj5uKoZJI0qJmw74YjWQPPyCpZ5v-L8ya03FXwK2C0Y/edit?usp=sharing
    """

    # whether the analect should be run locally only (no-op in OSS slice)
    local_only: bool = False
    namespace: Tuple[str, ...] | str = ()
    version: str | None = None

    @classmethod
    def display_name(cls) -> str:
        """
        Display name for the analect.
        """
        return cls.__name__

    @classmethod
    def namespace_id(cls) -> List[str]:
        """
        Namespace id for the analect.
        """
        ns = cls.namespace
        if isinstance(ns, str):
            return [ns]
        elif isinstance(ns, Sequence):
            return list(ns)
        else:
            raise ValueError(f"Invalid namespace: {ns}")

    @classmethod
    async def new_from_entry_input(cls, entry_input: EntryInput) -> "EntryAnalectMixin":
        """
        Subclass can override this method to construct a new instance of itself.
        By default, we simply use cls().
        """

        assert issubclass(cls, Analect)
        assert issubclass(
            cls.get_input_type(),
            EntryInput,
        ), f"{cls} must take EntryInput"
        assert issubclass(
            cls.get_output_type(),
            EntryOutput,
        ), f"{cls} must return EntryOutput"
        return cls()
