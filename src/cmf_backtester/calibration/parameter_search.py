from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import polars as pl

from cmf_backtester.backtest.engine import BacktestEngine
from cmf_backtester.calibration.volatility import rolling_volatility_time
from cmf_backtester.data.loaders import MarketDataArrays
from cmf_backtester.data.splitting import split_mask
from cmf_backtester.portfolio.metrics import summarize_performance
from cmf_backtester.strategies.avellaneda_stoikov import (
    AvellanedaStoikovStrategy,
    config_from_dict,
)
from cmf_backtester.utils.config import deep_update


def iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def score_metrics(metrics: dict[str, Any], inventory_penalty: float, drawdown_penalty: float) -> float:
    return float(
        metrics["final_pnl"]
        - inventory_penalty * metrics["avg_abs_inventory"]
        - drawdown_penalty * metrics["max_drawdown"]
    )


def run_validation_grid(
    base_config: dict[str, Any],
    market_data: MarketDataArrays,
    grid: dict[str, list[Any]],
    output_path: str | Path,
    inventory_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
) -> pl.DataFrame:
    mask = split_mask(market_data.split, "validation")
    validation_data = market_data.subset(mask)
    vol_cfg = base_config.get("volatility", {})
    sigma = rolling_volatility_time(
        validation_data.timestamps,
        validation_data.mid_ticks,
        float(vol_cfg.get("window_seconds", 300.0)),
        float(vol_cfg.get("floor_ticks_per_sqrt_second", 0.1)),
    )

    rows: list[dict[str, Any]] = []
    for params in iter_grid(grid):
        cfg = deep_update(base_config, {"strategy": params})
        strategy = AvellanedaStoikovStrategy(config_from_dict(cfg["strategy"]))
        runtime_mode = cfg.get("runtime", {}).get("mode", "fast_numba")
        execution_cfg = cfg.get("execution", {})
        result = BacktestEngine(
            validation_data,
            strategy,
            sigma,
            runtime_mode=runtime_mode,
            fill_mode=str(execution_cfg.get("fill_mode", "full")),
            fees_bps=float(execution_cfg.get("fees_bps", 0.0)),
        ).run()
        metrics = summarize_performance(result)
        metrics.update(params)
        metrics["score"] = score_metrics(metrics, inventory_penalty, drawdown_penalty)
        rows.append(metrics)

    df = pl.DataFrame(rows).sort("score", descending=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    return df
