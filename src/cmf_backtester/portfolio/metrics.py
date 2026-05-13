from __future__ import annotations

import numpy as np


def max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(pnl)
    drawdowns = running_max - pnl
    return float(np.max(drawdowns))


def summarize_performance(result: "BacktestResult") -> dict[str, float | int | str]:
    # Import lazily to avoid a circular import at module load time.
    from cmf_backtester.backtest.recorder import BacktestResult

    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    pnl = result.pnl
    inventory = result.inventory
    quoted_spread = result.ask_quote_ticks - result.bid_quote_ticks
    quoted_spread = quoted_spread[np.isfinite(quoted_spread)]
    pnl_diff = np.diff(pnl) if pnl.size > 1 else np.asarray([0.0])
    return {
        "strategy": result.strategy_name,
        "rows": int(pnl.size),
        "final_pnl": float(pnl[-1]) if pnl.size else 0.0,
        "pnl_volatility": float(np.std(pnl_diff)) if pnl_diff.size else 0.0,
        "max_drawdown": max_drawdown(pnl),
        "turnover": float(result.turnover[-1]) if result.turnover.size else 0.0,
        "fill_count": int(result.fill_count[-1]) if result.fill_count.size else 0,
        "buy_fills": int(result.buy_fills[-1]) if result.buy_fills.size else 0,
        "sell_fills": int(result.sell_fills[-1]) if result.sell_fills.size else 0,
        "final_inventory": float(inventory[-1]) if inventory.size else 0.0,
        "max_abs_inventory": float(np.max(np.abs(inventory))) if inventory.size else 0.0,
        "avg_abs_inventory": float(np.mean(np.abs(inventory))) if inventory.size else 0.0,
        "avg_quoted_spread_ticks": float(np.mean(quoted_spread)) if quoted_spread.size else 0.0,
    }


def summarize_performance_by_date(
    result: "BacktestResult",
    dates: np.ndarray,
) -> list[dict[str, float | int | str]]:
    if dates.shape[0] != result.pnl.shape[0]:
        raise ValueError("dates length must match result length")

    rows: list[dict[str, float | int | str]] = []
    for date in np.unique(dates):
        idx = np.where(dates == date)[0]
        if idx.size == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1])
        prev = start - 1
        pnl_base = float(result.pnl[prev]) if prev >= 0 else 0.0
        turnover_base = float(result.turnover[prev]) if prev >= 0 else 0.0
        fill_base = int(result.fill_count[prev]) if prev >= 0 else 0
        buy_base = int(result.buy_fills[prev]) if prev >= 0 else 0
        sell_base = int(result.sell_fills[prev]) if prev >= 0 else 0
        quoted_spread = result.ask_quote_ticks[idx] - result.bid_quote_ticks[idx]
        quoted_spread = quoted_spread[np.isfinite(quoted_spread)]
        inventory = result.inventory[idx]
        rows.append(
            {
                "strategy": result.strategy_name,
                "date": str(date),
                "rows": int(idx.size),
                "pnl_contribution": float(result.pnl[end] - pnl_base),
                "end_cumulative_pnl": float(result.pnl[end]),
                "turnover_contribution": float(result.turnover[end] - turnover_base),
                "fill_count": int(result.fill_count[end] - fill_base),
                "buy_fills": int(result.buy_fills[end] - buy_base),
                "sell_fills": int(result.sell_fills[end] - sell_base),
                "end_inventory": float(result.inventory[end]),
                "max_abs_inventory": float(np.max(np.abs(inventory))) if inventory.size else 0.0,
                "avg_abs_inventory": float(np.mean(np.abs(inventory))) if inventory.size else 0.0,
                "avg_quoted_spread_ticks": float(np.mean(quoted_spread)) if quoted_spread.size else 0.0,
            }
        )
    return rows
