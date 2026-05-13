from __future__ import annotations

import math

import numpy as np


def rolling_volatility_time(
    timestamps_us: np.ndarray,
    mid_ticks: np.ndarray,
    window_seconds: float,
    floor_ticks_per_sqrt_second: float = 0.0,
) -> np.ndarray:
    """Rolling realized volatility of mid changes in ticks per sqrt(second).

    Only past changes are used for index i. The first value is set to the floor.
    """
    n = len(timestamps_us)
    out = np.full(n, float(floor_ticks_per_sqrt_second), dtype=np.float64)
    if n <= 1:
        return out

    dt_seconds = np.diff(timestamps_us).astype(np.float64) / 1_000_000.0
    dm = np.diff(mid_ticks).astype(np.float64)
    variance_rate = np.zeros(n - 1, dtype=np.float64)
    valid = dt_seconds > 0
    variance_rate[valid] = (dm[valid] ** 2) / dt_seconds[valid]

    left = 0
    rolling_sum = 0.0
    for right in range(n - 1):
        rolling_sum += variance_rate[right]
        cutoff = timestamps_us[right + 1] - int(round(window_seconds * 1_000_000))
        while left <= right and timestamps_us[left + 1] < cutoff:
            rolling_sum -= variance_rate[left]
            left += 1
        count = right - left + 1
        vol = math.sqrt(max(rolling_sum / max(count, 1), 0.0))
        out[right + 1] = max(float(floor_ticks_per_sqrt_second), vol)
    return out

