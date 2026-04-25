# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Callable, Tuple, Type

from .manager import _parse_namespace, NamespaceId, register
from .mixin import EntryAnalectMixin


def _get_default_namespace(cls: Type[EntryAnalectMixin]) -> Tuple[str, ...]:
    namespace = cls.__module__.split(".")
    if namespace[0] == "confucius":
        namespace = namespace[1:]
    if namespace[0] == "analects":
        namespace = namespace[1:]
    # remove the last part of namespace, which is usually a module name
    if len(namespace) > 1:
        namespace = namespace[:-1]
    return tuple(namespace)


def public(
    *args: Type[EntryAnalectMixin],
    local_only: bool = False,
    namespace: NamespaceId = None,
    version: str | None = None,
) -> (
    Callable[[Type[EntryAnalectMixin]], Type[EntryAnalectMixin]]
    | Type[EntryAnalectMixin]
):
    """
    Decorator to make an EntryAnalectMixin public.
    When using this decorator, make sure that your Analect is actually included
    in the top-level __init__ target (https://fburl.com/code/e6lzn0s5), otherwise
    it still wouldn't be registered.
    (If you don't register it somewhere, the module simply wouldn't be shipped
    with the binary, and there is nothing we can do about it.)
    Args:
        *args: Optional. A single argument of type `type` representing the class to be decorated.
        local_only (bool, optional): Whether the EntryAnalectMixin should be marked as local only. Defaults to False.
        namespace (NamespaceId, optional): The namespace to register the EntryAnalectMixin to. Defaults to None.
        version (str, optional): The version of the EntryAnalectMixin. For UI display. Defaults to None.
    Usage Examples:
        @public(local_only=True)
        class MyClass(EntryAnalectMixin):
            pass

        @public
        class MyOtherClass(EntryAnalectMixin):
            pass
    """
    if args:
        (cls,) = args
        if isinstance(cls, type) and issubclass(cls, EntryAnalectMixin):
            # The decorator was used without arguments
            # pyre-ignore: Incompatible return type
            return public()(cls)
        else:
            raise TypeError(
                f"{cls} is not a type and a subclass of EntryAnalectMixin. Please use @public() instead of @public."
            )
    else:

        def decorator(cls: Type[EntryAnalectMixin]) -> Type[EntryAnalectMixin]:
            cls.local_only = local_only
            cls.namespace = (
                _get_default_namespace(cls)
                if namespace is None
                else _parse_namespace(namespace)
            )
            cls.version = version
            register(cls, override=True)
            return cls

        return decorator
