from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 1000
    n_steps: int = 1000
    dt: float = 0.01
    initial_mid: float = 100.0
    sigma: float = 1.0
    gamma: float = 0.1
    k: float = 1.5
    a: float = 1.0
    tau: float = 1.0
    order_size: float = 1.0
    seed: int = 7


def _simulate_path(cfg: MonteCarloConfig, rng: np.random.Generator, inventory_aware: bool) -> tuple[float, float, float, int]:
    mid = cfg.initial_mid
    cash = 0.0
    inventory = 0.0
    turnover = 0.0
    fills = 0
    base_spread = cfg.gamma * cfg.sigma**2 * cfg.tau + (2.0 / cfg.gamma) * np.log1p(
        cfg.gamma / cfg.k
    )

    for _ in range(cfg.n_steps):
        if inventory_aware:
            reservation = mid - inventory * cfg.gamma * cfg.sigma**2 * cfg.tau
        else:
            reservation = mid
        bid = reservation - base_spread / 2.0
        ask = reservation + base_spread / 2.0
        bid_distance = max(mid - bid, 0.0)
        ask_distance = max(ask - mid, 0.0)

        p_buy_fill = 1.0 - np.exp(-cfg.a * np.exp(-cfg.k * bid_distance) * cfg.dt)
        p_sell_fill = 1.0 - np.exp(-cfg.a * np.exp(-cfg.k * ask_distance) * cfg.dt)

        if rng.random() < p_buy_fill:
            value = bid * cfg.order_size
            cash -= value
            inventory += cfg.order_size
            turnover += abs(value)
            fills += 1
        if rng.random() < p_sell_fill:
            value = ask * cfg.order_size
            cash += value
            inventory -= cfg.order_size
            turnover += abs(value)
            fills += 1

        mid += cfg.sigma * np.sqrt(cfg.dt) * rng.normal()

    pnl = cash + inventory * mid
    return pnl, inventory, turnover, fills


def run_monte_carlo(cfg: MonteCarloConfig) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, float | int | str]] = []
    for strategy_name, inventory_aware in [
        ("as_inventory_aware", True),
        ("symmetric_mid_quotes", False),
    ]:
        for path in range(cfg.n_paths):
            pnl, inventory, turnover, fills = _simulate_path(cfg, rng, inventory_aware)
            rows.append(
                {
                    "strategy": strategy_name,
                    "path": path,
                    "pnl": pnl,
                    "final_inventory": inventory,
                    "turnover": turnover,
                    "fills": fills,
                }
            )
    return pl.DataFrame(rows)


def summarize_monte_carlo(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("strategy")
        .agg(
            pl.col("pnl").mean().alias("mean_pnl"),
            pl.col("pnl").std().alias("std_pnl"),
            pl.col("pnl").quantile(0.05).alias("p05_pnl"),
            pl.col("pnl").quantile(0.95).alias("p95_pnl"),
            pl.col("final_inventory").mean().alias("mean_final_inventory"),
            pl.col("final_inventory").std().alias("std_final_inventory"),
            pl.col("turnover").mean().alias("mean_turnover"),
            pl.col("fills").mean().alias("mean_fills"),
        )
        .sort("strategy")
    )


def write_monte_carlo_outputs(
    cfg: MonteCarloConfig,
    results_path: str | Path,
    summary_path: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    results = run_monte_carlo(cfg)
    summary = summarize_monte_carlo(results)
    results_path = Path(results_path)
    summary_path = Path(summary_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(results_path)
    summary.write_csv(summary_path)
    return results, summary

