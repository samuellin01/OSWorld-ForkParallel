# pyre-strict

from __future__ import annotations

import asyncio
import random
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Iterable, Tuple, TypeVar, Union

T = TypeVar("T")


class RetryableException(Exception):
    pass


RETRYABLE_CONNECTION_ERRS: tuple[type[Exception], ...] = (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    RetryableException,
)


def _compute_sleep_intervals(
    retries: int,
    sleep_time: float | None,
    sleep_time_intervals: list[float] | None,
    splay: float,
) -> Iterable[float]:
    """
    Compute sleep intervals with exponential backoff and jitter.

    Args:
        retries: Number of retry attempts to generate intervals for
        sleep_time: Base sleep time (defaults to 1.0 if None)
        sleep_time_intervals: Custom intervals (if provided, overrides calculation)
        splay: Jitter factor (adds ±splay/10 randomization to prevent thundering herd)

    Returns:
        List of sleep intervals for each retry attempt
    """
    if sleep_time_intervals:
        return sleep_time_intervals[:retries]
    base = sleep_time if sleep_time is not None else 1.0
    return [
        base * (2**i) * random.uniform(1 - splay / 10, 1 + splay / 10)
        for i in range(retries)
    ]


# Decorator compatible with existing call sites using **self.retryable_config.dict()
# Supported keys: retries, max_duration, splay, sleep_time, sleep_time_intervals
# Usage for async functions only in our codebase


def retryable(
    *,
    exceptions: Union[
        Tuple[type[BaseException], ...], Tuple[type[Exception], ...]
    ] = RETRYABLE_CONNECTION_ERRS,
    retries: int = 3,
    max_duration: float = 0.0,
    splay: float = 1.0,
    sleep_time: float | None = None,
    sleep_time_intervals: list[float] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    Retry decorator for async functions with exponential backoff and jitter.

    Performs 1 initial attempt + `retries` retry attempts (total: retries + 1 attempts).

    Args:
        exceptions: Tuple of exception types to retry on (default: connection errors)
        retries: Number of retry attempts after initial failure (default: 3)
        max_duration: Maximum total time in seconds (0 = no limit, default: 0.0)
        splay: Jitter factor for randomization (±10% by default, range: 0-10)
        sleep_time: Base sleep time in seconds (default: 1.0 if not specified)
        sleep_time_intervals: Custom list of sleep intervals (overrides other sleep settings)

    Returns:
        Decorated async function with retry logic

    Example:
        @retryable(retries=2, sleep_time=0.5)
        async def fetch_data():
            # Will attempt up to 3 times total (1 initial + 2 retries)
            pass
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            last_exc: BaseException | None = None

            # Try initial attempt
            try:
                return await func(*args, **kwargs)
            except exceptions as e:  # type: ignore[misc]
                last_exc = e

            # Retry attempts with delays
            for _attempt, delay in enumerate(
                _compute_sleep_intervals(
                    retries, sleep_time, sleep_time_intervals, splay
                ),
                1,
            ):
                # Check max duration before sleeping
                if max_duration and (time.time() - start) >= max_duration:
                    break

                if delay > 0:
                    await asyncio.sleep(delay)

                # Check max duration after sleep as well
                if max_duration and (time.time() - start) >= max_duration:
                    break

                try:
                    return await func(*args, **kwargs)
                except exceptions as e:  # type: ignore[misc]
                    last_exc = e
                    continue

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
