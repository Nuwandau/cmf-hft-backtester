from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import polars as pl

from cmf_backtester.liquidation.config import LiquidationEdaConfig
from cmf_backtester.liquidation.features import add_liquidation_features
from cmf_backtester.liquidation.io import collect_frame, scan_source
from cmf_backtester.liquidation.schema import (
    BINANCE_BBO,
    BINANCE_LIQUIDATIONS,
    BINANCE_TRADES,
    BYBIT_LIQUIDATIONS,
    SourceSpec,
)


PRICE_LOCATIONS = (
    "above_ask",
    "at_ask",
    "inside_spread",
    "at_bid",
    "below_bid",
    "outside_or_ambiguous",
    "missing_bbo",
)

PRESSURE_LABELS = {
    0: "downward_pressure",
    1: "no_pressure",
    2: "upward_pressure",
}
RELATION_LABELS = {
    0: "opposite_direction",
    1: "none",
    2: "same_direction_toxic_risk",
}
SIDE_LABELS = {
    0: "buy",
    1: "sell",
}
SIGNED_BUCKET_LABELS = {0: "zero"}
SIGNED_BUCKET_LABELS.update({idx + 1: f"pos_1e{idx}" for idx in range(10)})
SIGNED_BUCKET_LABELS.update({idx + 11: f"neg_1e{idx}" for idx in range(10)})


@dataclass
class FullEdaOutputs:
    tables: dict[str, pl.DataFrame]
    metadata: dict[str, Any]


def _utc_us(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def _date_start_us(value: date) -> int:
    return _utc_us(datetime(value.year, value.month, value.day))


def _date_end_us(value: date) -> int:
    return _date_start_us(value + timedelta(days=1)) - 1


def _date_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def _time_batches(start_us: int, end_us: int, batch_minutes: int) -> list[tuple[int, int]]:
    step = max(1, int(batch_minutes)) * 60 * 1_000_000
    batches: list[tuple[int, int]] = []
    current = start_us
    while current <= end_us:
        batch_end = min(end_us, current + step - 1)
        batches.append((current, batch_end))
        current = batch_end + 1
    return batches


def _effective_end_us(start_us: int, end_us: int, batch_minutes: int, max_batches: int) -> int:
    if max_batches <= 0:
        return end_us
    span = max_batches * max(1, int(batch_minutes)) * 60 * 1_000_000
    return min(end_us, start_us + span - 1)


def _split_dates(profile: str) -> list[tuple[str, date]]:
    train = [("train", d) for d in _date_range(date(2025, 12, 1), date(2026, 1, 31))]
    validation = [("validation", d) for d in _date_range(date(2026, 2, 1), date(2026, 2, 28))]
    if profile == "quick":
        return train[:1] + validation[:1]
    return train + validation


def _scan_between(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    start_us: int,
    end_us: int,
    columns: list[str],
) -> pl.DataFrame:
    return collect_frame(
        scan_source(config, spec)
        .filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= end_us))
        .select(columns)
        .sort("timestamp")
    )


def _side_sign(side: np.ndarray) -> np.ndarray:
    return np.where(side == "buy", 1.0, np.where(side == "sell", -1.0, np.nan))


def _split_from_date(day: date) -> str:
    if date(2025, 12, 1) <= day <= date(2026, 1, 31):
        return "train"
    if date(2026, 2, 1) <= day <= date(2026, 2, 28):
        return "validation"
    return "outside"


def _safe_idx(ts: np.ndarray, query: np.ndarray, *, include_same_timestamp: bool = True) -> np.ndarray:
    side = "right" if include_same_timestamp else "left"
    return np.searchsorted(ts, query, side=side) - 1


def _valid_asof(ts: np.ndarray, idx: np.ndarray, query: np.ndarray, tolerance_us: int) -> np.ndarray:
    valid = idx >= 0
    if not np.any(valid):
        return valid
    age = np.full(query.shape, np.iinfo(np.int64).max, dtype=np.int64)
    age[valid] = query[valid] - ts[idx[valid]]
    return valid & (age >= 0) & (age <= tolerance_us)


def _bbo_arrays(bbo: pl.DataFrame) -> dict[str, np.ndarray]:
    bid = bbo["bid_price"].to_numpy()
    ask = bbo["ask_price"].to_numpy()
    bid_amt = bbo["bid_amount"].to_numpy()
    ask_amt = bbo["ask_amount"].to_numpy()
    mid = (bid + ask) * 0.5
    spread = ask - bid
    denom = bid_amt + ask_amt
    imbalance = np.divide(bid_amt, denom, out=np.full_like(bid_amt, 0.5), where=denom != 0)
    return {
        "timestamp": bbo["timestamp"].to_numpy(),
        "bid": bid,
        "ask": ask,
        "bid_amount": bid_amt,
        "ask_amount": ask_amt,
        "mid": mid,
        "spread": spread,
        "imbalance": imbalance,
    }


def _add_weighted(agg: dict[tuple[Any, ...], dict[str, float]], key: tuple[Any, ...], pnl: np.ndarray, weight: np.ndarray) -> None:
    mask = np.isfinite(pnl) & np.isfinite(weight) & (weight > 0)
    state = agg[key]
    if not np.any(mask):
        return
    values = pnl[mask]
    weights = weight[mask]
    state["rows"] += float(values.size)
    state["pnl_sum"] += float(values.sum())
    state["weighted_pnl_sum"] += float(np.sum(values * weights))
    state["weight_sum"] += float(weights.sum())


def _agg_to_frame(
    agg: dict[tuple[Any, ...], dict[str, float]],
    key_cols: list[str],
    *,
    include_mean: bool = True,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, state in agg.items():
        weight_sum = state.get("weight_sum", 0.0)
        count = state.get("rows", 0.0)
        row = dict(zip(key_cols, key, strict=True))
        row["rows"] = int(count)
        row["clipped_turnover"] = weight_sum
        row["weighted_pnl_bps"] = state["weighted_pnl_sum"] / weight_sum if weight_sum else None
        if include_mean:
            row["mean_pnl_bps"] = state["pnl_sum"] / count if count else None
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _count_to_frame(agg: dict[tuple[Any, ...], int], key_cols: list[str]) -> pl.DataFrame:
    rows = [dict(zip(key_cols, key, strict=True), rows=int(value)) for key, value in agg.items()]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _location_from_arrays(
    price: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    valid: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    loc = np.full(price.shape, "missing_bbo", dtype=object)
    loc[valid & (price > ask + tolerance)] = "above_ask"
    loc[valid & (np.abs(price - ask) <= tolerance)] = "at_ask"
    loc[valid & (price < bid - tolerance)] = "below_bid"
    loc[valid & (np.abs(price - bid) <= tolerance)] = "at_bid"
    loc[valid & (price > bid) & (price < ask)] = "inside_spread"
    loc[valid & (loc == "missing_bbo")] = "outside_or_ambiguous"
    return loc


def _pressure_bucket(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, "none", dtype=object)
    out[values > 0] = "upward_pressure"
    out[values < 0] = "downward_pressure"
    out[np.abs(values) <= 1e-12] = "no_pressure"
    return out


def _pressure_relation(side_sign: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    product = side_sign * pressure
    out = np.full(product.shape, "none", dtype=object)
    out[product > 0] = "same_direction_toxic_risk"
    out[product < 0] = "opposite_direction"
    out[np.abs(product) <= 1e-12] = "none"
    return out


def _signed_log_bucket(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, "zero", dtype=object)
    abs_values = np.abs(values)
    nonzero = abs_values > 1e-12
    signs = np.where(values[nonzero] > 0, "pos", "neg")
    powers = np.floor(np.log10(abs_values[nonzero])).astype(int)
    powers = np.clip(powers, 0, 9)
    out[nonzero] = np.char.add(np.char.add(signs.astype(str), "_1e"), powers.astype(str))
    return out


def _signed_log_bucket_codes(values: np.ndarray) -> np.ndarray:
    codes = np.zeros(values.shape, dtype=np.int16)
    abs_values = np.abs(values)
    nonzero = abs_values > 1e-12
    if not np.any(nonzero):
        return codes
    powers = np.floor(np.log10(abs_values[nonzero])).astype(np.int16)
    powers = np.clip(powers, 0, 9)
    positive = values[nonzero] > 0
    codes[nonzero] = np.where(positive, powers + 1, powers + 11)
    return codes


def _add_weighted_code_groups(
    agg: dict[tuple[Any, ...], dict[str, float]],
    key_prefix: tuple[Any, ...],
    group_codes: np.ndarray,
    group_labels: dict[int, tuple[Any, ...]],
    pnl: np.ndarray,
    weight: np.ndarray,
    key_suffix: tuple[Any, ...] = (),
) -> None:
    valid = np.isfinite(pnl) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return
    codes = group_codes[valid].astype(np.int64, copy=False)
    values = pnl[valid]
    weights = weight[valid]
    max_code = int(codes.max(initial=0))
    counts = np.bincount(codes, minlength=max_code + 1)
    pnl_sums = np.bincount(codes, weights=values, minlength=max_code + 1)
    weight_sums = np.bincount(codes, weights=weights, minlength=max_code + 1)
    weighted_sums = np.bincount(codes, weights=values * weights, minlength=max_code + 1)
    for code, count in enumerate(counts):
        if count == 0 or code not in group_labels:
            continue
        state = agg[key_prefix + group_labels[code] + key_suffix]
        state["rows"] += float(count)
        state["pnl_sum"] += float(pnl_sums[code])
        state["weighted_pnl_sum"] += float(weighted_sums[code])
        state["weight_sum"] += float(weight_sums[code])


def _context_group_codes(side: np.ndarray, side_sign: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    side_code = np.where(side == "buy", 0, 1).astype(np.int16)
    pressure_sign = np.sign(pressure).astype(np.int16)
    pressure_code = pressure_sign + 1
    relation_sign = np.sign(side_sign * pressure_sign).astype(np.int16)
    relation_code = relation_sign + 1
    return (side_code * 9 + pressure_code * 3 + relation_code).astype(np.int16)


def _context_group_labels() -> dict[int, tuple[Any, ...]]:
    labels: dict[int, tuple[Any, ...]] = {}
    for side_code, side_label in SIDE_LABELS.items():
        for pressure_code, pressure_label in PRESSURE_LABELS.items():
            for relation_code, relation_label in RELATION_LABELS.items():
                code = side_code * 9 + pressure_code * 3 + relation_code
                labels[code] = (side_label, pressure_label, relation_label)
    return labels


def _nonlinear_group_codes(side: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    side_code = np.where(side == "buy", 0, 1).astype(np.int16)
    bucket_code = _signed_log_bucket_codes(pressure)
    return (side_code * len(SIGNED_BUCKET_LABELS) + bucket_code).astype(np.int16)


def _nonlinear_group_labels() -> dict[int, tuple[Any, ...]]:
    labels: dict[int, tuple[Any, ...]] = {}
    size = len(SIGNED_BUCKET_LABELS)
    for side_code, side_label in SIDE_LABELS.items():
        for bucket_code, bucket_label in SIGNED_BUCKET_LABELS.items():
            labels[side_code * size + bucket_code] = (side_label, bucket_label)
    return labels


def _load_liq_for_window(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    start_us: int,
    end_us: int,
    *,
    venue: str,
) -> dict[str, np.ndarray]:
    raw = _scan_between(
        config,
        spec,
        start_us,
        end_us,
        ["timestamp", "ticker", "side", "price", "amount"],
    )
    if raw.height == 0:
        return {"timestamp": np.array([], dtype=np.int64), "signed": np.array([], dtype=np.float64)}
    enriched = add_liquidation_features(raw, venue=venue, bybit_delay_us=config.bybit_delay_us)
    return {
        "timestamp": enriched["available_timestamp"].to_numpy(),
        "signed": enriched["signed_liquidation_notional"].to_numpy(),
        "side": enriched["side"].to_numpy(),
        "notional": enriched["notional"].to_numpy(),
    }


def _rolling_signed_pressure(
    event_ts: np.ndarray,
    liq_ts: np.ndarray,
    liq_signed: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    if event_ts.size == 0 or liq_ts.size == 0:
        return np.zeros(event_ts.shape, dtype=np.float64)
    order = np.argsort(liq_ts, kind="stable")
    ts = liq_ts[order]
    signed = liq_signed[order]
    cumsum = np.cumsum(signed)
    at_event = _safe_idx(ts, event_ts, include_same_timestamp=True)
    at_start = _safe_idx(ts, event_ts - window_seconds * 1_000_000, include_same_timestamp=True)
    result = np.zeros(event_ts.shape, dtype=np.float64)
    valid_event = at_event >= 0
    result[valid_event] += cumsum[at_event[valid_event]]
    valid_start = at_start >= 0
    result[valid_start] -= cumsum[at_start[valid_start]]
    return result


def _update_trade_and_markout_aggregates(
    *,
    config: LiquidationEdaConfig,
    symbol: str,
    split: str,
    day: date,
    trades: pl.DataFrame,
    bbo: pl.DataFrame,
    liq_by_venue: dict[str, dict[str, np.ndarray]],
    trade_summary: dict[tuple[Any, ...], dict[str, float]],
    price_locations: dict[tuple[Any, ...], int],
    asof_sensitivity: dict[tuple[Any, ...], int],
    markout_summary: dict[tuple[Any, ...], dict[str, float]],
    daily_markout: dict[tuple[Any, ...], dict[str, float]],
    context_agg: dict[tuple[Any, ...], dict[str, float]],
    nonlinear_agg: dict[tuple[Any, ...], dict[str, float]],
    response_agg: dict[tuple[Any, ...], dict[str, float]],
) -> None:
    if trades.height == 0 or bbo.height == 0:
        return
    bbo_arr = _bbo_arrays(bbo)
    bbo_ts = bbo_arr["timestamp"]
    if bbo_ts.size == 0:
        return

    trade_ts = trades["timestamp"].to_numpy()
    price = trades["price"].to_numpy()
    amount = trades["amount"].to_numpy()
    side = trades["side"].to_numpy()
    side_sign = _side_sign(side)
    notional = price * amount
    clipped = np.minimum(notional, 100_000.0)
    day_str = day.isoformat()

    for side_value in ("buy", "sell"):
        mask = side == side_value
        if not np.any(mask):
            continue
        key = (split, day_str, symbol, side_value)
        state = trade_summary[key]
        state["rows"] += float(np.sum(mask))
        state["amount_sum"] += float(np.sum(amount[mask]))
        state["notional_sum"] += float(np.sum(notional[mask]))
        state["clipped_turnover"] += float(np.sum(clipped[mask]))

    idx_same = _safe_idx(bbo_ts, trade_ts, include_same_timestamp=True)
    valid_same = _valid_asof(bbo_ts, idx_same, trade_ts, config.bbo_staleness_tolerance_us)
    idx_strict = _safe_idx(bbo_ts, trade_ts, include_same_timestamp=False)
    valid_strict = _valid_asof(bbo_ts, idx_strict, trade_ts, config.bbo_staleness_tolerance_us)

    bid_same = np.full(price.shape, np.nan)
    ask_same = np.full(price.shape, np.nan)
    bid_same[valid_same] = bbo_arr["bid"][idx_same[valid_same]]
    ask_same[valid_same] = bbo_arr["ask"][idx_same[valid_same]]
    locations_same = _location_from_arrays(price, bid_same, ask_same, valid_same)

    bid_strict = np.full(price.shape, np.nan)
    ask_strict = np.full(price.shape, np.nan)
    bid_strict[valid_strict] = bbo_arr["bid"][idx_strict[valid_strict]]
    ask_strict[valid_strict] = bbo_arr["ask"][idx_strict[valid_strict]]
    locations_strict = _location_from_arrays(price, bid_strict, ask_strict, valid_strict)

    for side_value in ("buy", "sell"):
        side_mask = side == side_value
        for location in PRICE_LOCATIONS:
            price_locations[(split, day_str, symbol, side_value, location)] += int(
                np.sum(side_mask & (locations_same == location))
            )
            asof_sensitivity[("same_timestamp_allowed", split, day_str, symbol, side_value, location)] += int(
                np.sum(side_mask & (locations_same == location))
            )
            asof_sensitivity[("strictly_previous_bbo", split, day_str, symbol, side_value, location)] += int(
                np.sum(side_mask & (locations_strict == location))
            )

    context_code_cache: dict[tuple[str, int], np.ndarray] = {}
    nonlinear_code_cache: dict[tuple[str, int], np.ndarray] = {}
    context_labels = _context_group_labels()
    nonlinear_labels = _nonlinear_group_labels()
    for venue, liq in liq_by_venue.items():
        for window in config.eda_curve_horizons_seconds:
            pressure = _rolling_signed_pressure(
                trade_ts,
                liq["timestamp"],
                liq["signed"],
                int(window),
            )
            key = (venue, int(window))
            context_code_cache[key] = _context_group_codes(side, side_sign, pressure)
            nonlinear_code_cache[key] = _nonlinear_group_codes(side, pressure)

    for horizon in config.eda_curve_horizons_seconds:
        target = trade_ts + horizon * 1_000_000
        future_idx = _safe_idx(bbo_ts, target, include_same_timestamp=True)
        valid_future = _valid_asof(bbo_ts, future_idx, target, config.bbo_staleness_tolerance_us)
        future_mid = np.full(price.shape, np.nan)
        future_mid[valid_future] = bbo_arr["mid"][future_idx[valid_future]]
        pnl = -side_sign * (future_mid - price) / price * 10_000.0 + config.maker_rebate_bps

        signed_flow = side_sign * notional
        valid_response = np.isfinite(pnl) & np.isfinite(signed_flow)
        resp_state = response_agg[(split, symbol, "binance_trade_flow", int(horizon))]
        if np.any(valid_response):
            resp_state["rows"] += float(np.sum(valid_response))
            resp_state["abs_flow_sum"] += float(np.sum(np.abs(signed_flow[valid_response])))
            resp_state["signed_response_sum"] += float(np.sum(pnl[valid_response] * signed_flow[valid_response]))
            resp_state["return_sum"] += float(np.sum(pnl[valid_response]))

        for side_value in ("buy", "sell"):
            mask = side == side_value
            if not np.any(mask):
                continue
            _add_weighted(
                markout_summary,
                (split, symbol, side_value, int(horizon)),
                pnl[mask],
                clipped[mask],
            )
            _add_weighted(
                daily_markout,
                (split, day_str, symbol, side_value, int(horizon)),
                pnl[mask],
                clipped[mask],
            )

        if int(horizon) not in config.task_horizons_seconds:
            continue
        for venue in ("binance", "bybit"):
            for window in config.eda_curve_horizons_seconds:
                cache_key = (venue, int(window))
                _add_weighted_code_groups(
                    context_agg,
                    (split, symbol),
                    context_code_cache[cache_key],
                    context_labels,
                    pnl,
                    clipped,
                    (venue, int(window), int(horizon)),
                )
                _add_weighted_code_groups(
                    nonlinear_agg,
                    (split, symbol),
                    nonlinear_code_cache[cache_key],
                    nonlinear_labels,
                    pnl,
                    clipped,
                    (venue, int(window), int(horizon)),
                )


def _update_bbo_aggregates(
    *,
    symbol: str,
    split: str,
    day: date,
    bbo: pl.DataFrame,
    bbo_quality: dict[tuple[Any, ...], dict[str, float]],
    bbo_ofi: dict[tuple[Any, ...], dict[str, float]],
    qi_counts: dict[tuple[Any, ...], dict[str, float]],
) -> None:
    if bbo.height < 2:
        return
    arr = _bbo_arrays(bbo)
    bid = arr["bid"]
    ask = arr["ask"]
    bid_amt = arr["bid_amount"]
    ask_amt = arr["ask_amount"]
    mid = arr["mid"]
    spread = arr["spread"]
    imbalance = arr["imbalance"]
    spread_bps = spread / mid * 10_000.0
    q_state = bbo_quality[(split, day.isoformat(), symbol)]
    q_state["rows"] += float(bbo.height)
    q_state["crossed_rows"] += float(np.sum(bid > ask))
    q_state["locked_rows"] += float(np.sum(bid == ask))
    q_state["nonpositive_bid_rows"] += float(np.sum(bid <= 0))
    q_state["nonpositive_ask_rows"] += float(np.sum(ask <= 0))
    q_state["negative_bid_amount_rows"] += float(np.sum(bid_amt < 0))
    q_state["negative_ask_amount_rows"] += float(np.sum(ask_amt < 0))
    q_state["spread_bps_sum"] += float(np.sum(spread_bps[np.isfinite(spread_bps)]))
    q_state["imbalance_sum"] += float(np.sum(imbalance[np.isfinite(imbalance)]))

    prev_bid = bid[:-1]
    cur_bid = bid[1:]
    prev_ask = ask[:-1]
    cur_ask = ask[1:]
    prev_bid_amt = bid_amt[:-1]
    cur_bid_amt = bid_amt[1:]
    prev_ask_amt = ask_amt[:-1]
    cur_ask_amt = ask_amt[1:]
    ofi = (
        np.where(cur_bid >= prev_bid, cur_bid_amt, 0.0)
        - np.where(cur_bid <= prev_bid, prev_bid_amt, 0.0)
        - np.where(cur_ask <= prev_ask, cur_ask_amt, 0.0)
        + np.where(cur_ask >= prev_ask, prev_ask_amt, 0.0)
    )
    next_return_bps = (mid[1:] - mid[:-1]) / mid[:-1] * 10_000.0
    valid = np.isfinite(ofi) & np.isfinite(next_return_bps)
    ofi_state = bbo_ofi[(split, symbol)]
    if np.any(valid):
        x = ofi[valid]
        y = next_return_bps[valid]
        ofi_state["rows"] += float(x.size)
        ofi_state["ofi_sum"] += float(np.sum(x))
        ofi_state["return_sum"] += float(np.sum(y))
        ofi_state["ofi_sq_sum"] += float(np.sum(x * x))
        ofi_state["return_sq_sum"] += float(np.sum(y * y))
        ofi_state["cross_sum"] += float(np.sum(x * y))

    buckets = np.floor(imbalance[:-1] * 10).astype(int)
    buckets = np.clip(buckets, 0, 9) + 1
    next_mid = mid[1:]
    cur_mid = mid[:-1]
    for bucket in range(1, 11):
        mask = buckets == bucket
        if not np.any(mask):
            continue
        state = qi_counts[(split, symbol, bucket)]
        state["rows"] += float(np.sum(mask))
        state["imbalance_sum"] += float(np.sum(imbalance[:-1][mask]))
        state["up_count"] += float(np.sum(next_mid[mask] > cur_mid[mask]))
        state["down_count"] += float(np.sum(next_mid[mask] < cur_mid[mask]))


def _update_liquidation_summary(
    *,
    symbol: str,
    split: str,
    day: date,
    venue: str,
    liq: dict[str, np.ndarray],
    summary: dict[tuple[Any, ...], dict[str, float]],
) -> None:
    if liq["timestamp"].size == 0:
        return
    start_us = _date_start_us(day)
    end_us = _date_end_us(day)
    in_day = (liq["timestamp"] >= start_us) & (liq["timestamp"] <= end_us)
    side = liq.get("side", np.array([], dtype=object))
    notional = liq.get("notional", np.array([], dtype=np.float64))
    for side_value in ("buy", "sell"):
        mask = in_day & (side == side_value)
        if not np.any(mask):
            continue
        state = summary[(split, day.isoformat(), venue, symbol, side_value)]
        state["rows"] += float(np.sum(mask))
        state["notional_sum"] += float(np.sum(notional[mask]))
        state["clipped_turnover"] += float(np.sum(np.minimum(notional[mask], 100_000.0)))


def _update_liquidation_response(
    *,
    symbol: str,
    venue: str,
    split: str,
    day: date,
    liq: dict[str, np.ndarray],
    bbo: pl.DataFrame,
    horizons: tuple[int, ...],
    tolerance_us: int,
    response_agg: dict[tuple[Any, ...], dict[str, float]],
) -> None:
    if liq["timestamp"].size == 0 or bbo.height == 0:
        return
    bbo_arr = _bbo_arrays(bbo)
    bbo_ts = bbo_arr["timestamp"]
    start_us = _date_start_us(day)
    end_us = _date_end_us(day)
    in_day = (liq["timestamp"] >= start_us) & (liq["timestamp"] <= end_us)
    event_ts = liq["timestamp"][in_day]
    signed = liq["signed"][in_day]
    if event_ts.size == 0:
        return
    event_idx = _safe_idx(bbo_ts, event_ts, include_same_timestamp=True)
    event_valid = _valid_asof(bbo_ts, event_idx, event_ts, tolerance_us)
    event_mid = np.full(event_ts.shape, np.nan)
    event_mid[event_valid] = bbo_arr["mid"][event_idx[event_valid]]
    for horizon in horizons:
        target = event_ts + horizon * 1_000_000
        future_idx = _safe_idx(bbo_ts, target, include_same_timestamp=True)
        valid_future = _valid_asof(bbo_ts, future_idx, target, tolerance_us)
        future_mid = np.full(event_ts.shape, np.nan)
        future_mid[valid_future] = bbo_arr["mid"][future_idx[valid_future]]
        ret = (future_mid - event_mid) / event_mid * 10_000.0
        valid = np.isfinite(ret) & np.isfinite(signed)
        if not np.any(valid):
            continue
        state = response_agg[(split, symbol, f"{venue}_liquidation_flow", int(horizon))]
        state["rows"] += float(np.sum(valid))
        state["abs_flow_sum"] += float(np.sum(np.abs(signed[valid])))
        state["signed_response_sum"] += float(np.sum(ret[valid] * signed[valid]))
        state["return_sum"] += float(np.sum(ret[valid]))


def _update_event_study(
    *,
    symbol: str,
    venue: str,
    split: str,
    day: date,
    liq: dict[str, np.ndarray],
    bbo: pl.DataFrame,
    offsets_seconds: tuple[int, ...],
    tolerance_us: int,
    event_study: dict[tuple[Any, ...], dict[str, float]],
) -> None:
    if liq["timestamp"].size == 0 or bbo.height == 0:
        return
    bbo_arr = _bbo_arrays(bbo)
    bbo_ts = bbo_arr["timestamp"]
    event_ts = liq["timestamp"]
    start_us = _date_start_us(day)
    end_us = _date_end_us(day)
    in_day = (event_ts >= start_us) & (event_ts <= end_us)
    event_ts = event_ts[in_day]
    side = liq.get("side", np.array([], dtype=object))[in_day]
    if event_ts.size == 0:
        return
    base_idx = _safe_idx(bbo_ts, event_ts, include_same_timestamp=True)
    base_valid = _valid_asof(bbo_ts, base_idx, event_ts, tolerance_us)
    base_mid = np.full(event_ts.shape, np.nan)
    base_mid[base_valid] = bbo_arr["mid"][base_idx[base_valid]]
    for offset in offsets_seconds:
        query = event_ts + offset * 1_000_000
        idx = _safe_idx(bbo_ts, query, include_same_timestamp=True)
        valid = _valid_asof(bbo_ts, idx, query, tolerance_us) & np.isfinite(base_mid)
        if not np.any(valid):
            continue
        ret = np.full(event_ts.shape, np.nan)
        ret[valid] = (bbo_arr["mid"][idx[valid]] - base_mid[valid]) / base_mid[valid] * 10_000.0
        for side_value in ("buy", "sell"):
            mask = valid & (side == side_value)
            if not np.any(mask):
                continue
            state = event_study[(split, day.isoformat(), symbol, venue, side_value, int(offset))]
            values = ret[mask]
            state["rows"] += float(values.size)
            state["return_sum"] += float(np.sum(values))
            state["return_sq_sum"] += float(np.sum(values * values))


def _response_to_frame(response_agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, symbol, flow_type, horizon), state in response_agg.items():
        rows_count = state.get("rows", 0.0)
        abs_flow = state.get("abs_flow_sum", 0.0)
        rows.append(
            {
                "symbol": symbol,
                "split": split,
                "flow_type": flow_type,
                "horizon_seconds": horizon,
                "rows": int(rows_count),
                "response_bps": state["signed_response_sum"] / abs_flow if abs_flow else None,
                "mean_return_bps": state["return_sum"] / rows_count if rows_count else None,
                "abs_signed_flow_musd": abs_flow / 1_000_000.0,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _trade_summary_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, state in agg.items():
        split, day, symbol, side = key
        rows_count = state.get("rows", 0.0)
        rows.append(
            {
                "split": split,
                "date": day,
                "symbol": symbol,
                "side": side,
                "rows": int(rows_count),
                "mean_amount": state["amount_sum"] / rows_count if rows_count else None,
                "mean_notional": state["notional_sum"] / rows_count if rows_count else None,
                "clipped_turnover": state["clipped_turnover"],
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _liquidation_summary_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, state in agg.items():
        split, day, venue, symbol, side = key
        rows_count = state.get("rows", 0.0)
        rows.append(
            {
                "split": split,
                "date": day,
                "venue": venue,
                "symbol": symbol,
                "side": side,
                "rows": int(rows_count),
                "mean_notional": state["notional_sum"] / rows_count if rows_count else None,
                "notional_sum": state["notional_sum"],
                "clipped_turnover": state["clipped_turnover"],
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _bbo_quality_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, day, symbol), state in agg.items():
        rows_count = state.get("rows", 0.0)
        rows.append(
            {
                "split": split,
                "date": day,
                "symbol": symbol,
                "rows": int(rows_count),
                "crossed_rows": int(state["crossed_rows"]),
                "locked_rows": int(state["locked_rows"]),
                "nonpositive_bid_rows": int(state["nonpositive_bid_rows"]),
                "nonpositive_ask_rows": int(state["nonpositive_ask_rows"]),
                "negative_bid_amount_rows": int(state["negative_bid_amount_rows"]),
                "negative_ask_amount_rows": int(state["negative_ask_amount_rows"]),
                "mean_spread_bps": state["spread_bps_sum"] / rows_count if rows_count else None,
                "mean_queue_imbalance": state["imbalance_sum"] / rows_count if rows_count else None,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _bbo_ofi_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, symbol), state in agg.items():
        n = state.get("rows", 0.0)
        if n <= 1:
            corr = None
        else:
            sx = state["ofi_sum"]
            sy = state["return_sum"]
            sxx = state["ofi_sq_sum"]
            syy = state["return_sq_sum"]
            sxy = state["cross_sum"]
            denom = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
            corr = (n * sxy - sx * sy) / denom if denom > 0 else None
        rows.append(
            {
                "split": split,
                "symbol": symbol,
                "rows": int(n),
                "mean_ofi": state["ofi_sum"] / n if n else None,
                "mean_next_return_bps": state["return_sum"] / n if n else None,
                "corr_ofi_next_return_bps": corr,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _qi_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, symbol, bucket), state in agg.items():
        n = state.get("rows", 0.0)
        up_count = state["up_count"]
        down_count = state["down_count"]
        moved_count = up_count + down_count
        rows.append(
            {
                "split": split,
                "symbol": symbol,
                "imbalance_bucket": int(bucket),
                "rows": int(n),
                "mean_queue_imbalance": state["imbalance_sum"] / n if n else None,
                "prob_next_event_up": up_count / n if n else None,
                "prob_next_event_down": down_count / n if n else None,
                "no_mid_move_share": 1.0 - moved_count / n if n else None,
                "prob_next_move_up": up_count / moved_count if moved_count else None,
                "prob_next_move_down": down_count / moved_count if moved_count else None,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _event_study_to_frame(agg: dict[tuple[Any, ...], dict[str, float]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, state in agg.items():
        split, day, symbol, venue, side, offset = key
        n = state.get("rows", 0.0)
        mean = state["return_sum"] / n if n else None
        variance = state["return_sq_sum"] / n - mean * mean if n and mean is not None else None
        rows.append(
            {
                "split": split,
                "date": day,
                "symbol": symbol,
                "venue": venue,
                "side": side,
                "offset_seconds": int(offset),
                "rows": int(n),
                "mean_return_bps": mean,
                "std_return_bps": float(np.sqrt(max(variance, 0.0))) if variance is not None else None,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def compute_full_data_eda(
    config: LiquidationEdaConfig,
    specs: list[SourceSpec],
) -> FullEdaOutputs:
    by_source_symbol = {(spec.source, spec.symbol): spec for spec in specs}
    trade_summary: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    liquidation_summary: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bbo_quality: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bbo_ofi: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    qi_counts: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    price_locations: dict[tuple[Any, ...], int] = defaultdict(int)
    asof_sensitivity: dict[tuple[Any, ...], int] = defaultdict(int)
    markout_summary: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    daily_markout: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    context_agg: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    nonlinear_agg: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    response_agg: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    event_study: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    max_horizon = max(config.eda_curve_horizons_seconds)
    max_context_window = max(config.eda_curve_horizons_seconds)
    offsets = (-300, -120, -60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60, 120, 300)
    processed_trade_rows = 0
    processed_bbo_rows = 0
    processed_trade_batches = 0
    batch_minutes = config.active_profile.full_data_batch_minutes
    max_batches_per_date = config.active_profile.max_full_data_batches_per_date

    for symbol in config.symbols:
        trade_spec = by_source_symbol[(BINANCE_TRADES, symbol)]
        bbo_spec = by_source_symbol[(BINANCE_BBO, symbol)]
        bin_liq_spec = by_source_symbol[(BINANCE_LIQUIDATIONS, symbol)]
        bybit_liq_spec = by_source_symbol[(BYBIT_LIQUIDATIONS, symbol)]
        for split, day in _split_dates(config.profile):
            start_us = _date_start_us(day)
            end_us = _date_end_us(day)
            effective_end_us = _effective_end_us(
                start_us,
                end_us,
                batch_minutes,
                max_batches_per_date,
            )
            bbo_start = start_us - max(abs(min(offsets)), max_context_window) * 1_000_000 - config.bbo_staleness_tolerance_us
            bbo_end = effective_end_us + max(max_horizon, max(offsets)) * 1_000_000 + config.bbo_staleness_tolerance_us
            bbo = _scan_between(
                config,
                bbo_spec,
                bbo_start,
                bbo_end,
                ["timestamp", "ticker", "bid_price", "bid_amount", "ask_price", "ask_amount"],
            )
            bbo_day = bbo.filter(
                (pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= effective_end_us)
            )
            processed_bbo_rows += bbo_day.height
            _update_bbo_aggregates(
                symbol=symbol,
                split=split,
                day=day,
                bbo=bbo_day,
                bbo_quality=bbo_quality,
                bbo_ofi=bbo_ofi,
                qi_counts=qi_counts,
            )

            liq_start = start_us - max_context_window * 1_000_000 - config.bybit_delay_us
            liq_end = effective_end_us + config.bybit_delay_us
            bin_liq = _load_liq_for_window(config, bin_liq_spec, liq_start, liq_end, venue="binance")
            bybit_liq = _load_liq_for_window(config, bybit_liq_spec, liq_start, liq_end, venue="bybit")
            liq_by_venue = {"binance": bin_liq, "bybit": bybit_liq}
            _update_liquidation_summary(
                symbol=symbol,
                split=split,
                day=day,
                venue="binance",
                liq=bin_liq,
                summary=liquidation_summary,
            )
            _update_liquidation_summary(
                symbol=symbol,
                split=split,
                day=day,
                venue="bybit",
                liq=bybit_liq,
                summary=liquidation_summary,
            )
            day_trade_rows = 0
            for batch_start, batch_end in _time_batches(start_us, effective_end_us, batch_minutes):
                trades = _scan_between(
                    config,
                    trade_spec,
                    batch_start,
                    batch_end,
                    ["timestamp", "ticker", "side", "price", "amount"],
                )
                processed_trade_batches += 1
                processed_trade_rows += trades.height
                day_trade_rows += trades.height
                if trades.height:
                    _update_trade_and_markout_aggregates(
                        config=config,
                        symbol=symbol,
                        split=split,
                        day=day,
                        trades=trades,
                        bbo=bbo,
                        liq_by_venue=liq_by_venue,
                        trade_summary=trade_summary,
                        price_locations=price_locations,
                        asof_sensitivity=asof_sensitivity,
                        markout_summary=markout_summary,
                        daily_markout=daily_markout,
                        context_agg=context_agg,
                        nonlinear_agg=nonlinear_agg,
                        response_agg=response_agg,
                    )
                del trades
            _update_liquidation_response(
                symbol=symbol,
                venue="binance",
                split=split,
                day=day,
                liq=bin_liq,
                bbo=bbo,
                horizons=config.eda_curve_horizons_seconds,
                tolerance_us=config.bbo_staleness_tolerance_us,
                response_agg=response_agg,
            )
            _update_liquidation_response(
                symbol=symbol,
                venue="bybit",
                split=split,
                day=day,
                liq=bybit_liq,
                bbo=bbo,
                horizons=config.eda_curve_horizons_seconds,
                tolerance_us=config.bbo_staleness_tolerance_us,
                response_agg=response_agg,
            )
            _update_event_study(
                symbol=symbol,
                venue="binance",
                split=split,
                day=day,
                liq=bin_liq,
                bbo=bbo,
                offsets_seconds=offsets,
                tolerance_us=config.bbo_staleness_tolerance_us,
                event_study=event_study,
            )
            _update_event_study(
                symbol=symbol,
                venue="bybit",
                split=split,
                day=day,
                liq=bybit_liq,
                bbo=bbo,
                offsets_seconds=offsets,
                tolerance_us=config.bbo_staleness_tolerance_us,
                event_study=event_study,
            )
            print(
                "[liquidation-eda] full-data chunk "
                f"{symbol} {split} {day.isoformat()} "
                f"bbo_rows={bbo_day.height} trade_rows={day_trade_rows} "
                f"batches={len(_time_batches(start_us, effective_end_us, batch_minutes))}",
                flush=True,
            )
            del bbo, bbo_day, bin_liq, bybit_liq, liq_by_venue

    price_location_df = _count_to_frame(
        price_locations,
        ["split", "date", "symbol", "side", "price_location"],
    )
    if price_location_df.height:
        price_location_df = price_location_df.with_columns(
            (
                pl.col("rows")
                / pl.col("rows").sum().over(["split", "date", "symbol", "side"])
            ).alias("share_within_side")
        )
    asof_df = _count_to_frame(
        asof_sensitivity,
        ["asof_mode", "split", "date", "symbol", "side", "price_location"],
    )
    if asof_df.height:
        asof_df = asof_df.with_columns(
            (
                pl.col("rows")
                / pl.col("rows").sum().over(["asof_mode", "split", "date", "symbol", "side"])
            ).alias("share_within_side")
        )

    tables = {
        "trade_summary": _trade_summary_to_frame(trade_summary),
        "liquidation_summary": _liquidation_summary_to_frame(liquidation_summary),
        "bbo_quality": _bbo_quality_to_frame(bbo_quality),
        "bbo_ofi_summary": _bbo_ofi_to_frame(bbo_ofi),
        "queue_imbalance_next_move": _qi_to_frame(qi_counts),
        "trade_price_location_summary": price_location_df,
        "asof_sensitivity": asof_df,
        "full_markout_summary": _agg_to_frame(
            markout_summary,
            ["split", "symbol", "side", "horizon_seconds"],
        ),
        "full_daily_weighted_markout": _agg_to_frame(
            daily_markout,
            ["split", "date", "symbol", "side", "horizon_seconds"],
        ),
        "full_markout_by_liquidation_context": _agg_to_frame(
            context_agg,
            [
                "split",
                "symbol",
                "side",
                "liq_pressure_bucket",
                "maker_vs_liq_pressure",
                "venue",
                "window_seconds",
                "horizon_seconds",
            ],
        ),
        "nonlinear_flow_response": _agg_to_frame(
            nonlinear_agg,
            [
                "split",
                "symbol",
                "side",
                "signed_liq_bucket",
                "venue",
                "window_seconds",
                "horizon_seconds",
            ],
        ),
        "signed_flow_response_functions": _response_to_frame(response_agg),
        "event_study_summary": _event_study_to_frame(event_study),
    }
    ordered_context_cols = [
        "split",
        "symbol",
        "side",
        "venue",
        "window_seconds",
        "liq_pressure_bucket",
        "maker_vs_liq_pressure",
        "horizon_seconds",
        "rows",
        "clipped_turnover",
        "weighted_pnl_bps",
        "mean_pnl_bps",
    ]
    ordered_nonlinear_cols = [
        "split",
        "symbol",
        "side",
        "venue",
        "window_seconds",
        "signed_liq_bucket",
        "horizon_seconds",
        "rows",
        "clipped_turnover",
        "weighted_pnl_bps",
        "mean_pnl_bps",
    ]
    for name, columns in [
        ("full_markout_by_liquidation_context", ordered_context_cols),
        ("nonlinear_flow_response", ordered_nonlinear_cols),
    ]:
        if tables[name].height:
            tables[name] = tables[name].select([col for col in columns if col in tables[name].columns])
    metadata = {
        "processed_trade_rows_full": processed_trade_rows,
        "processed_bbo_rows_full": processed_bbo_rows,
        "processed_trade_batches_full": processed_trade_batches,
        "full_data_batch_minutes": batch_minutes,
        "full_data_max_batches_per_date": max_batches_per_date,
        "full_data_dates_processed": len(_split_dates(config.profile)),
        "full_data_mode": config.profile == "full",
    }
    return FullEdaOutputs(tables=tables, metadata=metadata)
