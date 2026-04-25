# pyre-strict

from inspect import isawaitable
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def run_validator(
    validator: Callable[..., Awaitable[T] | T], *args: Any, **kwargs: Any
) -> T:
    """Run validator on value and assert that it returns the same value."""
    result = validator(*args, **kwargs)
    if isawaitable(result):
        return await result
    else:
        return result
