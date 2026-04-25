# pyre-strict

from __future__ import annotations


def get_human_delta(seconds: float) -> str:
    """
    Return a human-readable delta like '2d 3h', '4h 12m', '5m 10s', or '45s'.
    Rounds down for larger units and includes the next smaller unit when useful.
    """
    if seconds < 0:
        seconds = 0.0

    s = int(seconds)
    if s < 60:
        return f"{s}s"

    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m" + (f" {s}s" if s else "")

    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h" + (f" {m}m" if m else "")

    d, h = divmod(h, 24)
    return f"{d}d" + (f" {h}h" if h else "")
