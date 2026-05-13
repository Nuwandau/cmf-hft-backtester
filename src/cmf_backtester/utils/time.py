from __future__ import annotations

import datetime as dt


def timestamp_us_to_date(timestamp_us: int) -> str:
    """Convert Unix microseconds to an ISO UTC date string."""
    return dt.datetime.fromtimestamp(timestamp_us / 1_000_000, dt.UTC).date().isoformat()


def seconds_to_us(seconds: float) -> int:
    return int(round(seconds * 1_000_000))

