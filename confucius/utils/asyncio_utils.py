# pyre-strict
from __future__ import annotations

import asyncio


async def cancel(fut: asyncio.Future) -> None:
    """
    Cancel a Future or Task and wait until cancellation completes.

    Behavior:
    - If the input is already done(), this is a no-op.
    - If this coroutine is itself cancelled while waiting, it preserves the
      cancellation contract by re-raising CancelledError after ensuring the
      target task has been driven to a terminal state.
    - If the task raises an exception during cancellation, the exception is
      re-raised to the caller.
    - If the task completes normally instead of cancelling, InvalidStateError
      is raised.
    """
    if fut.done():
        return  # nothing to do
    fut.cancel()
    exc: asyncio.CancelledError | None = None
    while not fut.done():
        shielded = asyncio.shield(fut)
        try:
            await asyncio.wait([shielded])
        except asyncio.CancelledError as ex:  # preserve cancellation contract
            exc = ex
        finally:
            # Ensure we retrieve the result/exception to avoid log noise from asyncio
            if (
                shielded.done()
                and not shielded.cancelled()
                and not shielded.exception()
            ):
                shielded.result()
    if fut.cancelled():
        if exc is None:
            return
        # we were cancelled also, so honor the contract
        raise exc from None
    # Some exception thrown during cancellation
    ex = fut.exception()
    if ex is not None:
        raise ex from None
    # fut finished instead of cancelled
    raise asyncio.InvalidStateError(
        f"task didn't raise CancelledError on cancel: {fut} had result {fut.result()}"
    )
