from datetime import date

import numpy as np

from cmf_backtester.liquidation.full_eda import (
    _context_group_codes,
    _date_end_us,
    _date_start_us,
    _rolling_signed_pressure,
    _safe_idx,
    _signed_log_bucket_codes,
    _time_batches,
    _valid_asof,
)


def test_time_batches_cover_range_without_overlap() -> None:
    start = _date_start_us(date(2026, 2, 1))
    end = start + 10 * 60 * 1_000_000 - 1
    batches = _time_batches(start, end, batch_minutes=3)

    assert batches[0][0] == start
    assert batches[-1][1] == end
    assert all(left[1] + 1 == right[0] for left, right in zip(batches, batches[1:]))


def test_safe_idx_same_timestamp_vs_strict_previous() -> None:
    ts = np.array([10, 20, 30], dtype=np.int64)
    query = np.array([5, 10, 25, 30], dtype=np.int64)

    assert _safe_idx(ts, query, include_same_timestamp=True).tolist() == [-1, 0, 1, 2]
    assert _safe_idx(ts, query, include_same_timestamp=False).tolist() == [-1, -1, 1, 1]


def test_valid_asof_enforces_backward_tolerance() -> None:
    ts = np.array([100, 200], dtype=np.int64)
    query = np.array([150, 400], dtype=np.int64)
    idx = _safe_idx(ts, query, include_same_timestamp=True)

    assert _valid_asof(ts, idx, query, tolerance_us=100).tolist() == [True, False]


def test_rolling_signed_pressure_uses_past_window_only() -> None:
    events = np.array([10_000_000, 20_000_000], dtype=np.int64)
    liq_ts = np.array([5_000_000, 9_000_000, 15_000_000, 25_000_000], dtype=np.int64)
    signed = np.array([1.0, 2.0, -4.0, 100.0])

    pressure = _rolling_signed_pressure(events, liq_ts, signed, window_seconds=10)

    assert pressure.tolist() == [3.0, -4.0]


def test_context_codes_and_signed_log_buckets_are_stable() -> None:
    side = np.array(["buy", "sell", "buy"], dtype=object)
    side_sign = np.array([1.0, -1.0, 1.0])
    pressure = np.array([100.0, 0.0, -10_000.0])

    codes = _context_group_codes(side, side_sign, pressure)
    bucket_codes = _signed_log_bucket_codes(pressure)

    assert codes.tolist() == [8, 13, 0]
    assert bucket_codes.tolist() == [3, 0, 15]


def test_date_us_helpers_bound_utc_day() -> None:
    start = _date_start_us(date(2026, 2, 1))
    end = _date_end_us(date(2026, 2, 1))

    assert end - start + 1 == 24 * 60 * 60 * 1_000_000
