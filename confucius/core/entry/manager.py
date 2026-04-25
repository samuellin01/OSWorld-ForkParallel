# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, Field, PrivateAttr

from .mixin import EntryAnalectMixin


GLOBAL_NAMESPACE = "__global__"


NamespaceId = str | Sequence[str] | None


def _parse_namespace(namespace: NamespaceId) -> tuple[str, ...]:
    if namespace is None:
        return ()
    elif isinstance(namespace, str):
        return (namespace,)
    elif isinstance(namespace, Sequence):
        return tuple(namespace)
    else:
        raise ValueError(f"Invalid namespace: {namespace}")


class Namespace(BaseModel):
    """
    The namespace of the entry analects
    """

    id: list[str] = Field([], description="index coordinate of the namespace")
    name: str = Field(..., description="name of the namespace")
    children: dict[str, Namespace] = Field(
        default_factory=dict, description="sub namespaces"
    )
    _entries: dict[str, type[EntryAnalectMixin]] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def get_namespace(
        self,
        indices: str | tuple[str, ...],
        *,
        create: bool = False,
    ) -> Namespace:
        """
        Get the namespace at a specific index coordinate
        Args:
            indices: the index coordinate of the namespace, can be a string or a list of strings, if None, the namespace will be searched in the global namespace
            create: whether to create the namespace if it does not exist
        Returns:
            the namespace
        """
        namespace = _parse_namespace(indices)

        if len(namespace) == 0:
            return self

        child_ns_name = namespace[0]
        if child_ns_name not in self.children:
            if not create:
                raise KeyError(
                    f'Namespace "{child_ns_name}" does not exist in "{self.name}"'
                )
            self.children[child_ns_name] = Namespace(
                name=child_ns_name, id=self.id + [child_ns_name]
            )

        return self.children[child_ns_name].get_namespace(namespace[1:], create=create)

    def __getitem__(self, indices: str | tuple[str, ...]) -> Namespace:
        return self.get_namespace(indices)

    def __contains__(self, item: Namespace | type[EntryAnalectMixin]) -> bool:
        if isinstance(item, Namespace):
            return self.has_namespace(item)
        elif isinstance(item, type) and issubclass(item, EntryAnalectMixin):
            return self.has_entry(item)
        else:
            raise TypeError(f"Invalid item type: {type(item)}")

    def has_namespace(self, namespace: Namespace) -> bool:
        if namespace.name in self.children:
            return True

        return any(child.has_namespace(namespace) for child in self.children.values())

    def has_entry(self, entry: type[EntryAnalectMixin]) -> bool:
        if entry.display_name() in self._entries:
            return True

        return any(child.has_entry(entry) for child in self.children.values())

    @property
    def entries(self) -> Iterable[type[EntryAnalectMixin]]:
        """
        Get all entry analects in the namespace
        """
        yield from self._entries.values()

        for child in self.children.values():
            yield from child.entries

    def add_entry(
        self, entry: type[EntryAnalectMixin], *, override: bool = False
    ) -> None:
        """
        Add an entry analect to the namespace
        Args:
            entry: the entry analect to add
            override: whether to override the existing entry analect
        """
        if entry.display_name() in self._entries and not override:
            raise KeyError(
                f'Entry analect "{entry.display_name()}" already exists in "{self.name}". If you want to override it, set "override" to True. Or using a different display_name() for the entry analect or select a different namespace.'
            )
        self._entries[entry.display_name()] = entry

    def remove_entry(self, entry: type[EntryAnalectMixin]) -> None:
        """
        Remove an entry analect from the namespace
        Args:
            entry: the entry analect to remove
        """
        self._entries.pop(entry.display_name(), None)

    def get_entry(self, display_name: str) -> type[EntryAnalectMixin]:
        """
        Get the entry analect by display name
        Args:
            display_name: the display name of the entry analect
        Returns:
            the entry analect

        Raises:
            KeyError: if the entry analect does not exist
        """
        if display_name in self._entries:
            return self._entries[display_name]

        for sub_ns in self.children.values():
            try:
                return sub_ns.get_entry(display_name)
            except KeyError:
                pass

        raise KeyError(
            f'Entry analect "{display_name}" does not exist in "{self.name}"'
        )


class EntryManager:
    """
    Manager for entry analects
    """

    def __init__(self) -> None:
        self._root = Namespace(name=GLOBAL_NAMESPACE)

    def register(
        self, entry: type[EntryAnalectMixin], *, override: bool = False
    ) -> None:
        """
        Register an entry analect to the entry manager
        Args:
            entry: the entry analect to register
            override: whether to override the existing entry analect
        """
        self._root.get_namespace(
            _parse_namespace(entry.namespace), create=True
        ).add_entry(entry, override=override)

    def unregister(self, entry: type[EntryAnalectMixin]) -> None:
        """
        Unregister an entry analect to the entry manager
        Args:
            entry: the entry analect to unregister
        """
        self._root.get_namespace(_parse_namespace(entry.namespace)).remove_entry(entry)

    def get_entries(
        self,
        namespace: NamespaceId = None,
    ) -> Iterable[type[EntryAnalectMixin]]:
        """
        Get all the entry analects in a specific namespace
        Args:
            namespace: the namespace to get the entry analects, can be a string or a list of strings, if None, the entry analects will be searched in the global namespace
        Returns:
            the entry analects
        Raises:
            KeyError: if the namespace does not exist
        """
        yield from self._root[_parse_namespace(namespace)].entries

    def get_entry(
        self,
        display_name: str,
        namespace: NamespaceId = None,
    ) -> type[EntryAnalectMixin]:
        """
        Get an entry analect by name
        Args:
            display_name: the display name of the entry analect
            namespace: the namespace of the entry analect, can be a string or a list of strings, if None, the entry analect will be searched in the global namespace
        Returns:
            the entry analect
        Raises:
            KeyError: if the entry analect does not exist
        """
        return self._root[_parse_namespace(namespace)].get_entry(display_name)

    def get_namespace(self, namespace: NamespaceId) -> Namespace:
        """
        Get the namespace by name
        Args:
            namespace: the namespace to get, can be a string or a list of strings
        Returns:
            the namespace
        Raises:
            KeyError: if the namespace does not exist
        """
        return self._root[_parse_namespace(namespace)]


_ENTRY_MGR: EntryManager = EntryManager()


def register(entry: type[EntryAnalectMixin], *, override: bool = False) -> None:
    """
    See EntryManager.register
    """
    _ENTRY_MGR.register(entry, override=override)


def get_entries(
    namespace: NamespaceId = None,
) -> Iterable[type[EntryAnalectMixin]]:
    """
    See EntryManager.get_entries
    """
    return _ENTRY_MGR.get_entries(namespace)


def get_entry(
    display_name: str,
    namespace: NamespaceId = None,
) -> type[EntryAnalectMixin]:
    """
    See EntryManager.get_entry
    """
    return _ENTRY_MGR.get_entry(display_name, namespace)


def get_namespace(namespace: NamespaceId) -> Namespace:
    """
    See EntryManager.get_namespace
    """
    return _ENTRY_MGR.get_namespace(namespace)
