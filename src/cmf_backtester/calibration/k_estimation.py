from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _crossing_probability_kernel(
    best_bid_ticks: np.ndarray,
    best_ask_ticks: np.ndarray,
    distance: int,
    horizon_events: int,
) -> float:
    n = best_bid_ticks.shape[0]
    if n <= horizon_events + 1:
        return 0.0
    crosses = 0
    trials = 0
    for i in range(n - horizon_events):
        bid_quote = best_bid_ticks[i] - distance
        ask_quote = best_ask_ticks[i] + distance
        future_ask_min = best_ask_ticks[i + 1]
        future_bid_max = best_bid_ticks[i + 1]
        for j in range(i + 1, i + 1 + horizon_events):
            if best_ask_ticks[j] < future_ask_min:
                future_ask_min = best_ask_ticks[j]
            if best_bid_ticks[j] > future_bid_max:
                future_bid_max = best_bid_ticks[j]
        if future_ask_min <= bid_quote or future_bid_max >= ask_quote:
            crosses += 1
        trials += 1
    return crosses / max(trials, 1)


def estimate_crossing_probabilities(
    best_bid_ticks: np.ndarray,
    best_ask_ticks: np.ndarray,
    distances_ticks: list[int],
    horizon_events: int,
) -> dict[int, float]:
    """Estimate future crossing probability for hypothetical passive quotes.

    For each event i, place a bid at bid_i - delta and an ask at ask_i + delta.
    A crossing occurs if a future ask crosses the bid or a future bid crosses the ask.
    """
    n = len(best_bid_ticks)
    out: dict[int, float] = {}
    if n <= horizon_events + 1:
        return {d: 0.0 for d in distances_ticks}

    bid = best_bid_ticks.astype(np.int64)
    ask = best_ask_ticks.astype(np.int64)
    for distance in distances_ticks:
        out[distance] = float(
            _crossing_probability_kernel(
                bid,
                ask,
                int(distance),
                int(horizon_events),
            )
        )
    return out


def fit_exponential_k(distances_ticks: np.ndarray, probabilities: np.ndarray, horizon_seconds: float) -> float:
    """Fit log(lambda(delta)) = a - k delta from crossing probabilities."""
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    intensities = -np.log(1.0 - clipped) / max(horizon_seconds, 1e-12)
    y = np.log(np.clip(intensities, 1e-12, None))
    x = distances_ticks.astype(np.float64)
    slope, _intercept = np.polyfit(x, y, 1)
    return float(max(1e-12, -slope))
