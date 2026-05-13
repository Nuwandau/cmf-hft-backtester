from __future__ import annotations

import math


def floor_to_int(value: float) -> int:
    return int(math.floor(value))


def ceil_to_int(value: float) -> int:
    return int(math.ceil(value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator

