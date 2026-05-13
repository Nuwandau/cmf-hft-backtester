from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def run_as_crossing_kernel(
    timestamps: np.ndarray,
    best_bid_ticks: np.ndarray,
    best_ask_ticks: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
    mid_ticks: np.ndarray,
    sigma_ticks_per_sqrt_second: np.ndarray,
    microprice_adjustment_ticks: np.ndarray,
    tick_size: float,
    gamma: float,
    k: float,
    tau_seconds: float,
    order_size: float,
    max_inventory: float,
    inventory_risk_unit: float,
    min_spread_ticks: int,
    quote_refresh_seconds: float,
    post_only: bool,
    fill_mode_code: int,
    fees_bps: float,
):
    """Numba-accelerated AS crossing backtest kernel.

    The kernel models one active bid and one active ask. That matches the strategy design:
    every quote refresh cancels previous quotes and places at most one order on each side.
    """
    n = timestamps.shape[0]
    pnl = np.zeros(n, dtype=np.float64)
    cash_arr = np.zeros(n, dtype=np.float64)
    inventory_arr = np.zeros(n, dtype=np.float64)
    turnover_arr = np.zeros(n, dtype=np.float64)
    fill_count_arr = np.zeros(n, dtype=np.int64)
    buy_fills_arr = np.zeros(n, dtype=np.int64)
    sell_fills_arr = np.zeros(n, dtype=np.int64)
    reservation_ticks_arr = np.empty(n, dtype=np.float64)
    bid_quote_ticks_arr = np.empty(n, dtype=np.float64)
    ask_quote_ticks_arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        reservation_ticks_arr[i] = np.nan
        bid_quote_ticks_arr[i] = np.nan
        ask_quote_ticks_arr[i] = np.nan

    cash = 0.0
    inventory = 0.0
    turnover = 0.0
    fill_count = 0
    buy_fills = 0
    sell_fills = 0

    has_bid = False
    has_ask = False
    active_bid_ticks = 0
    active_ask_ticks = 0
    active_bid_remaining = 0.0
    active_ask_remaining = 0.0
    bid_created_ts = 0
    ask_created_ts = 0
    last_quote_ts = -9223372036854775807
    refresh_us = int(round(quote_refresh_seconds * 1_000_000.0))

    safe_gamma = gamma
    if safe_gamma < 1e-18:
        safe_gamma = 1e-18
    safe_k = k
    if safe_k < 1e-18:
        safe_k = 1e-18
    safe_inventory_risk_unit = inventory_risk_unit
    if safe_inventory_risk_unit < 1e-18:
        safe_inventory_risk_unit = order_size
    if safe_inventory_risk_unit < 1e-18:
        safe_inventory_risk_unit = 1e-18

    for i in range(n):
        ts = timestamps[i]

        if has_bid and bid_created_ts < ts and best_ask_ticks[i] <= active_bid_ticks:
            fill_qty = active_bid_remaining
            if fill_mode_code == 1 and ask_size[i] < fill_qty:
                fill_qty = ask_size[i]
            if fill_qty > 0.0:
                value = active_bid_ticks * tick_size * fill_qty
                fee = abs(value) * fees_bps / 10_000.0
                cash -= value + fee
                inventory += fill_qty
                turnover += abs(value)
                fill_count += 1
                buy_fills += 1
                active_bid_remaining -= fill_qty
                if active_bid_remaining <= 1e-12:
                    has_bid = False
                    active_bid_remaining = 0.0

        if has_ask and ask_created_ts < ts and best_bid_ticks[i] >= active_ask_ticks:
            fill_qty = active_ask_remaining
            if fill_mode_code == 1 and bid_size[i] < fill_qty:
                fill_qty = bid_size[i]
            if fill_qty > 0.0:
                value = active_ask_ticks * tick_size * fill_qty
                fee = abs(value) * fees_bps / 10_000.0
                cash += value - fee
                inventory -= fill_qty
                turnover += abs(value)
                fill_count += 1
                sell_fills += 1
                active_ask_remaining -= fill_qty
                if active_ask_remaining <= 1e-12:
                    has_ask = False
                    active_ask_remaining = 0.0

        variance_horizon = (
            sigma_ticks_per_sqrt_second[i] * sigma_ticks_per_sqrt_second[i] * tau_seconds
        )
        reference_ticks = mid_ticks[i] + microprice_adjustment_ticks[i]
        model_inventory = inventory / safe_inventory_risk_unit
        reservation_ticks = reference_ticks - model_inventory * safe_gamma * variance_horizon
        reservation_ticks_arr[i] = reservation_ticks

        if last_quote_ts == -9223372036854775807 or ts - last_quote_ts >= refresh_us:
            has_bid = False
            has_ask = False
            active_bid_remaining = 0.0
            active_ask_remaining = 0.0

            total_spread_ticks = safe_gamma * variance_horizon + (
                2.0 / safe_gamma
            ) * np.log1p(safe_gamma / safe_k)
            if total_spread_ticks < min_spread_ticks:
                total_spread_ticks = float(min_spread_ticks)

            bid_ticks = int(np.floor(reservation_ticks - total_spread_ticks / 2.0))
            ask_ticks = int(np.ceil(reservation_ticks + total_spread_ticks / 2.0))
            if ask_ticks - bid_ticks < min_spread_ticks:
                ask_ticks = bid_ticks + min_spread_ticks

            if post_only:
                if bid_ticks > best_bid_ticks[i]:
                    bid_ticks = best_bid_ticks[i]
                if ask_ticks < best_ask_ticks[i]:
                    ask_ticks = best_ask_ticks[i]
                if ask_ticks - bid_ticks < min_spread_ticks:
                    ask_ticks = bid_ticks + min_spread_ticks

            if inventory < max_inventory:
                has_bid = True
                active_bid_ticks = bid_ticks
                active_bid_remaining = order_size
                bid_created_ts = ts
            if inventory > -max_inventory:
                has_ask = True
                active_ask_ticks = ask_ticks
                active_ask_remaining = order_size
                ask_created_ts = ts
            last_quote_ts = ts

        pnl[i] = cash + inventory * mid_ticks[i] * tick_size
        cash_arr[i] = cash
        inventory_arr[i] = inventory
        turnover_arr[i] = turnover
        fill_count_arr[i] = fill_count
        buy_fills_arr[i] = buy_fills
        sell_fills_arr[i] = sell_fills
        if has_bid:
            bid_quote_ticks_arr[i] = active_bid_ticks
        if has_ask:
            ask_quote_ticks_arr[i] = active_ask_ticks

    return (
        pnl,
        cash_arr,
        inventory_arr,
        turnover_arr,
        fill_count_arr,
        buy_fills_arr,
        sell_fills_arr,
        reservation_ticks_arr,
        bid_quote_ticks_arr,
        ask_quote_ticks_arr,
    )
