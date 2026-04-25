# pyre-strict

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

T = TypeVar("T")


def await_sync(awaitable: Coroutine[Any, Any, T]) -> T:
    """
    Run an awaitable from sync context and return its result.

    - If called from a running event loop, run the coroutine in a dedicated thread
      with a fresh event loop to avoid "event loop is running" errors.
    - Otherwise, use asyncio.run() directly.
    """

    try:
        asyncio.get_running_loop()
        # Already in an event loop, run in a separate thread with a new loop
        result_holder: dict[str, Any] = {}
        exc_holder: dict[str, Exception] = {}

        def _runner() -> None:
            try:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    result_holder["value"] = new_loop.run_until_complete(awaitable)
                finally:
                    new_loop.close()
            except Exception as e:
                exc_holder["exc"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()

        if "exc" in exc_holder:
            raise exc_holder["exc"]
        return result_holder["value"]  # pyre-ignore[16]
    except RuntimeError:
        # No running loop
        return asyncio.run(awaitable)


async def convert_to_async(
    func: Callable[..., T | Awaitable[T]], *args: Any, **kwargs: Any
) -> T:
    """
    Call a possibly-sync function in an async-friendly way.

    - If func returns an awaitable, await it and return the result.
    - Otherwise, run it in a background thread and return the result.
    """
    res = func(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res  # type: ignore[return-value]
    # Run blocking call off the event loop
    return await asyncio.to_thread(lambda: res)
