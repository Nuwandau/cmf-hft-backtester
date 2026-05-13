from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def compute_mid(best_bid: float, best_ask: float) -> float:
    return 0.5 * (best_bid + best_ask)


def compute_spread(best_bid: float, best_ask: float) -> float:
    return best_ask - best_bid


def compute_imbalance(bid_size: float, ask_size: float) -> float:
    denom = bid_size + ask_size
    if denom <= 0:
        return 0.5
    return bid_size / denom


def estimate_tick_size_from_prices(prices: Iterable[float], scale: int = 10_000_000) -> float:
    """Estimate a robust minimum positive price difference from observed prices."""
    scaled = sorted({int(round(float(price) * scale)) for price in prices if price is not None})
    diffs = [b - a for a, b in zip(scaled, scaled[1:]) if b > a]
    if not diffs:
        raise ValueError("Cannot estimate tick size from fewer than two distinct prices")
    return min(diffs) / scale


def price_to_ticks(price: float, tick_size: float) -> int:
    return int(round(price / tick_size))


def floor_price_to_ticks(price_ticks: float) -> int:
    return int(math.floor(price_ticks))


def ceil_price_to_ticks(price_ticks: float) -> int:
    return int(math.ceil(price_ticks))


def imbalance_bucket(imbalance: float, n_buckets: int) -> int:
    """Return zero-based bucket for imbalance in [0, 1]."""
    if not np.isfinite(imbalance):
        return n_buckets // 2
    bucket = int(math.ceil(float(imbalance) * n_buckets)) - 1
    return min(n_buckets - 1, max(0, bucket))


def spread_state(spread_ticks: int, max_spread_state_ticks: int) -> int:
    """Return zero-based spread state. Last state is the tail state."""
    if spread_ticks <= 1:
        return 0
    if spread_ticks > max_spread_state_ticks:
        return max_spread_state_ticks
    return int(spread_ticks - 1)


def build_state_id(
    spread_ticks: int,
    imbalance: float,
    n_imbalance_buckets: int,
    max_spread_state_ticks: int,
) -> int:
    return (
        spread_state(spread_ticks, max_spread_state_ticks) * n_imbalance_buckets
        + imbalance_bucket(imbalance, n_imbalance_buckets)
    )

