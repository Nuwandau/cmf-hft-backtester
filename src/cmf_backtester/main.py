from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cmf_backtester.backtest.engine import BacktestEngine
from cmf_backtester.calibration.k_estimation import (
    estimate_crossing_probabilities,
    fit_exponential_k,
)
from cmf_backtester.calibration.parameter_search import run_validation_grid
from cmf_backtester.calibration.volatility import rolling_volatility_time
from cmf_backtester.data.loaders import load_processed_arrays
from cmf_backtester.data.preprocessing import (
    audit_processed_lob_by_date,
    audit_processed_lob,
    create_sample_data,
    estimate_tick_size_from_lob,
    preprocess_lob_l1,
)
from cmf_backtester.data.splitting import split_mask
from cmf_backtester.market.microprice import MicropriceEstimator
from cmf_backtester.portfolio.metrics import summarize_performance, summarize_performance_by_date
from cmf_backtester.strategies.avellaneda_stoikov import (
    AvellanedaStoikovStrategy,
    config_from_dict,
)
from cmf_backtester.strategies.avellaneda_stoikov_microprice import (
    AvellanedaStoikovMicropriceStrategy,
)
from cmf_backtester.utils.config import load_config


def _load_arrays_from_config(config: dict[str, Any]):
    data_cfg = config["data"]
    return load_processed_arrays(data_cfg["processed_lob_path"], float(data_cfg["tick_size"]))


def _sigma_from_config(config: dict[str, Any], arrays):
    vol_cfg = config.get("volatility", {})
    return rolling_volatility_time(
        arrays.timestamps,
        arrays.mid_ticks,
        float(vol_cfg.get("window_seconds", 300.0)),
        float(vol_cfg.get("floor_ticks_per_sqrt_second", 0.1)),
    )


def _strategy_from_config(config: dict[str, Any]):
    strategy_cfg = config["strategy"]
    as_cfg = config_from_dict(strategy_cfg)
    name = strategy_cfg.get("name", "avellaneda_stoikov_mid")
    if name == "avellaneda_stoikov_microprice":
        return AvellanedaStoikovMicropriceStrategy(as_cfg)
    if name == "avellaneda_stoikov_mid":
        return AvellanedaStoikovStrategy(as_cfg)
    raise ValueError(f"Unsupported strategy: {name}")


def _run_backtest_from_config(config: dict[str, Any]):
    arrays = _load_arrays_from_config(config)
    split = config["data"].get("split", "test")
    arrays = arrays.subset(split_mask(arrays.split, split))
    sigma = _sigma_from_config(config, arrays)
    strategy = _strategy_from_config(config)
    micro_adj = None
    if strategy.name == "avellaneda_stoikov_microprice":
        estimator_path = Path(config["microprice"]["estimator_path"])
        if not estimator_path.exists():
            raise FileNotFoundError(
                f"Microprice estimator not found: {estimator_path}. Run fit-microprice first."
            )
        estimator = MicropriceEstimator.load(estimator_path)
        micro_adj = estimator.predict_adjustments(arrays.spread_ticks, arrays.imbalance)

    runtime_mode = config.get("runtime", {}).get("mode", "fast_numba")
    execution_cfg = config.get("execution", {})
    result = BacktestEngine(
        arrays,
        strategy,
        sigma,
        micro_adj,
        runtime_mode=runtime_mode,
        fill_mode=str(execution_cfg.get("fill_mode", "full")),
        fees_bps=float(execution_cfg.get("fees_bps", 0.0)),
    ).run()
    metrics = summarize_performance(result)
    metrics["split"] = split
    metrics["runtime_mode"] = runtime_mode
    metrics["fill_mode"] = str(execution_cfg.get("fill_mode", "full"))
    metrics["fees_bps"] = float(execution_cfg.get("fees_bps", 0.0))
    return result, metrics


def cmd_preprocess_lob(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_cfg = config["data"]
    split_cfg = config["splits"]
    tick_size = float(data_cfg.get("tick_size") or estimate_tick_size_from_lob(data_cfg["raw_lob_path"]))
    preprocess_lob_l1(
        data_cfg["raw_lob_path"],
        data_cfg["processed_lob_path"],
        tick_size,
        split_cfg["train_dates"],
        split_cfg["validation_dates"],
        split_cfg["test_dates"],
    )
    create_sample_data(data_cfg["processed_lob_path"], data_cfg["sample_lob_path"], args.sample_rows)
    audit = audit_processed_lob(data_cfg["processed_lob_path"])
    out = Path("reports/tables/data_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{k: str(v) for k, v in audit.items()}]).write_csv(out)
    by_date_out = Path("reports/tables/data_audit_by_date.csv")
    audit_processed_lob_by_date(data_cfg["processed_lob_path"]).write_csv(by_date_out)
    print(f"Wrote {data_cfg['processed_lob_path']}")
    print(f"Wrote {data_cfg['sample_lob_path']}")
    print(f"Wrote {out}")
    print(f"Wrote {by_date_out}")


def cmd_audit_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    audit = audit_processed_lob(config["data"]["processed_lob_path"])
    for key, value in audit.items():
        print(f"{key}: {value}")


def cmd_fit_microprice(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    arrays = _load_arrays_from_config(config)
    train = arrays.subset(split_mask(arrays.split, "train"))
    mp_cfg = config["microprice"]
    estimator = MicropriceEstimator(
        n_imbalance_buckets=int(mp_cfg.get("imbalance_buckets", 10)),
        max_spread_state_ticks=int(mp_cfg.get("max_spread_state_ticks", 10)),
        max_mid_move_ticks=float(mp_cfg.get("max_mid_move_ticks", 1.0)),
        min_state_count=int(mp_cfg.get("min_state_count", 50)),
        max_iterations=int(mp_cfg.get("max_iterations", 1000)),
        tolerance=float(mp_cfg.get("tolerance", 1e-10)),
    ).fit(train)
    estimator.save(mp_cfg["estimator_path"])
    table_path = Path("reports/tables/microprice_state_diagnostics.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(estimator.adjustment_table()).write_csv(table_path)
    from cmf_backtester.reporting.plots import plot_microprice_adjustment

    plot_microprice_adjustment(estimator, "reports/figures/microprice_adjustment_by_imbalance.png")
    print(f"Wrote {mp_cfg['estimator_path']}")
    print(f"Wrote {table_path}")
    print(estimator.diagnostics)


def _fit_microprice_for_config(
    config: dict[str, Any],
    max_mid_move_ticks: float | None = None,
    estimator_path: str | Path | None = None,
) -> MicropriceEstimator:
    arrays = _load_arrays_from_config(config)
    train = arrays.subset(split_mask(arrays.split, "train"))
    mp_cfg = config["microprice"]
    estimator = MicropriceEstimator(
        n_imbalance_buckets=int(mp_cfg.get("imbalance_buckets", 10)),
        max_spread_state_ticks=int(mp_cfg.get("max_spread_state_ticks", 10)),
        max_mid_move_ticks=float(
            mp_cfg.get("max_mid_move_ticks", 1.0)
            if max_mid_move_ticks is None
            else max_mid_move_ticks
        ),
        min_state_count=int(mp_cfg.get("min_state_count", 50)),
        max_iterations=int(mp_cfg.get("max_iterations", 1000)),
        tolerance=float(mp_cfg.get("tolerance", 1e-10)),
    ).fit(train)
    if estimator_path is not None:
        estimator.save(estimator_path)
    return estimator


def _float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def cmd_run_microprice_move_sensitivity(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    values = _parse_csv_numbers(args.max_mid_moves, float)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    estimator_dir = Path(args.estimator_dir)
    estimator_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for value in values:
        estimator_path = estimator_dir / f"microprice_estimator_maxmove_{_float_label(value)}.npz"
        estimator = _fit_microprice_for_config(config, value, estimator_path)
        if estimator.diagnostics is None:
            raise RuntimeError("Microprice fitting did not produce diagnostics")
        diagnostics = estimator.diagnostics
        accepted_raw_transitions = diagnostics.n_observations / 2.0
        raw_transition_count = accepted_raw_transitions + diagnostics.n_filtered_transitions
        filtered_share = (
            diagnostics.n_filtered_transitions / raw_transition_count
            if raw_transition_count > 0
            else 0.0
        )

        for split in ["validation", "test"]:
            run_config = copy.deepcopy(config)
            run_config["data"]["split"] = split
            run_config["microprice"]["estimator_path"] = str(estimator_path)
            _result, metrics = _run_backtest_from_config(run_config)
            score = float(
                metrics["final_pnl"]
                - args.drawdown_penalty * metrics["max_drawdown"]
                - args.inventory_penalty * metrics["avg_abs_inventory"]
            )
            rows.append(
                {
                    "max_mid_move_ticks": float(value),
                    "split": split,
                    "score": score,
                    "final_pnl": metrics["final_pnl"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "fill_count": metrics["fill_count"],
                    "final_inventory": metrics["final_inventory"],
                    "max_abs_inventory": metrics["max_abs_inventory"],
                    "avg_abs_inventory": metrics["avg_abs_inventory"],
                    "avg_quoted_spread_ticks": metrics["avg_quoted_spread_ticks"],
                    "estimator_path": str(estimator_path),
                    "accepted_raw_transitions": accepted_raw_transitions,
                    "filtered_transitions": diagnostics.n_filtered_transitions,
                    "filtered_share": filtered_share,
                    "max_abs_adjustment_ticks": diagnostics.max_abs_adjustment_ticks,
                    "median_state_count": diagnostics.median_state_count,
                    "min_state_count": diagnostics.min_state_count,
                    "max_state_count": diagnostics.max_state_count,
                    "converged": diagnostics.converged,
                    "iterations": diagnostics.iterations,
                }
            )

    df = (
        pl.DataFrame(rows)
        .with_columns(
            pl.when(pl.col("split") == "validation")
            .then(0)
            .otherwise(1)
            .alias("_split_order")
        )
        .sort(["_split_order", "score"], descending=[False, True])
        .drop("_split_order")
    )
    df.write_csv(output)
    print(df)
    print(f"Wrote {output}")


def cmd_run_quote_refresh_sensitivity(args: argparse.Namespace) -> None:
    mid_config = load_config(args.mid_config)
    micro_config = load_config(args.micro_config)
    values = _parse_csv_numbers(args.refresh_seconds, float)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not Path(micro_config["microprice"]["estimator_path"]).exists():
        _fit_microprice_for_config(micro_config, estimator_path=micro_config["microprice"]["estimator_path"])

    rows: list[dict[str, Any]] = []
    for base_config in [mid_config, micro_config]:
        for value in values:
            for split in ["validation", "test"]:
                run_config = copy.deepcopy(base_config)
                run_config["data"]["split"] = split
                run_config["strategy"]["quote_refresh_seconds"] = float(value)
                _result, metrics = _run_backtest_from_config(run_config)
                score = float(
                    metrics["final_pnl"]
                    - args.drawdown_penalty * metrics["max_drawdown"]
                    - args.inventory_penalty * metrics["avg_abs_inventory"]
                )
                rows.append(
                    {
                        "strategy": metrics["strategy"],
                        "quote_refresh_seconds": float(value),
                        "split": split,
                        "score": score,
                        "final_pnl": metrics["final_pnl"],
                        "max_drawdown": metrics["max_drawdown"],
                        "turnover": metrics["turnover"],
                        "fill_count": metrics["fill_count"],
                        "final_inventory": metrics["final_inventory"],
                        "max_abs_inventory": metrics["max_abs_inventory"],
                        "avg_abs_inventory": metrics["avg_abs_inventory"],
                        "avg_quoted_spread_ticks": metrics["avg_quoted_spread_ticks"],
                    }
                )

    df = (
        pl.DataFrame(rows)
        .with_columns(
            pl.when(pl.col("split") == "validation")
            .then(0)
            .otherwise(1)
            .alias("_split_order")
        )
        .sort(["strategy", "_split_order", "score"], descending=[False, False, True])
        .drop("_split_order")
    )
    df.write_csv(output)
    from cmf_backtester.reporting.plots import plot_quote_refresh_sensitivity

    plot_quote_refresh_sensitivity(df, "reports/figures/quote_refresh_sensitivity.png")
    print(df)
    print(f"Wrote {output}")
    print("Wrote reports/figures/quote_refresh_sensitivity.png")


def cmd_run_volatility_sensitivity(args: argparse.Namespace) -> None:
    mid_config = load_config(args.mid_config)
    micro_config = load_config(args.micro_config)
    windows = _parse_csv_numbers(args.window_seconds, float)
    floors = _parse_csv_numbers(args.floors, float)
    multipliers = _parse_csv_numbers(args.multipliers, float)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not Path(micro_config["microprice"]["estimator_path"]).exists():
        _fit_microprice_for_config(micro_config, estimator_path=micro_config["microprice"]["estimator_path"])

    rows: list[dict[str, Any]] = []
    for base_config in [mid_config, micro_config]:
        arrays_all = _load_arrays_from_config(base_config)
        strategy_name = base_config["strategy"]["name"]
        estimator = None
        if strategy_name == "avellaneda_stoikov_microprice":
            estimator = MicropriceEstimator.load(base_config["microprice"]["estimator_path"])
        for window_seconds in windows:
            for floor in floors:
                for multiplier in multipliers:
                    for split in ["validation", "test"]:
                        arrays = arrays_all.subset(split_mask(arrays_all.split, split))
                        sigma = rolling_volatility_time(
                            arrays.timestamps,
                            arrays.mid_ticks,
                            window_seconds,
                            floor,
                        ) * multiplier
                        strategy = _strategy_from_config(base_config)
                        micro_adj = None
                        if estimator is not None:
                            micro_adj = estimator.predict_adjustments(
                                arrays.spread_ticks, arrays.imbalance
                            )
                        execution_cfg = base_config.get("execution", {})
                        result = BacktestEngine(
                            arrays,
                            strategy,
                            sigma,
                            micro_adj,
                            runtime_mode=base_config.get("runtime", {}).get("mode", "fast_numba"),
                            fill_mode=str(execution_cfg.get("fill_mode", "full")),
                            fees_bps=float(execution_cfg.get("fees_bps", 0.0)),
                        ).run()
                        metrics = summarize_performance(result)
                        score = float(
                            metrics["final_pnl"]
                            - args.drawdown_penalty * metrics["max_drawdown"]
                            - args.inventory_penalty * metrics["avg_abs_inventory"]
                        )
                        rows.append(
                            {
                                "strategy": metrics["strategy"],
                                "split": split,
                                "vol_window_seconds": float(window_seconds),
                                "vol_floor": float(floor),
                                "sigma_multiplier": float(multiplier),
                                "score": score,
                                "final_pnl": metrics["final_pnl"],
                                "max_drawdown": metrics["max_drawdown"],
                                "turnover": metrics["turnover"],
                                "fill_count": metrics["fill_count"],
                                "final_inventory": metrics["final_inventory"],
                                "max_abs_inventory": metrics["max_abs_inventory"],
                                "avg_abs_inventory": metrics["avg_abs_inventory"],
                                "avg_quoted_spread_ticks": metrics["avg_quoted_spread_ticks"],
                            }
                        )

    df = (
        pl.DataFrame(rows)
        .with_columns(
            pl.when(pl.col("split") == "validation")
            .then(0)
            .otherwise(1)
            .alias("_split_order")
        )
        .sort(["strategy", "_split_order", "score"], descending=[False, False, True])
        .drop("_split_order")
    )
    df.write_csv(output)
    from cmf_backtester.reporting.plots import plot_volatility_sensitivity

    plot_volatility_sensitivity(df, "reports/figures/volatility_sensitivity.png")
    print(df.head(20))
    print(f"Wrote {output}")
    print("Wrote reports/figures/volatility_sensitivity.png")


def cmd_diagnose_microprice_signal(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    arrays = _load_arrays_from_config(config)
    estimator_path = Path(config["microprice"]["estimator_path"])
    if not estimator_path.exists():
        _fit_microprice_for_config(config, estimator_path=estimator_path)
    estimator = MicropriceEstimator.load(estimator_path)

    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        data = arrays.subset(split_mask(arrays.split, split))
        if len(data) < 2:
            continue
        adjustment = estimator.predict_adjustments(data.spread_ticks[:-1], data.imbalance[:-1])
        next_mid_move = data.mid_ticks[1:] - data.mid_ticks[:-1]
        local_mask = np.abs(next_mid_move) <= estimator.max_mid_move_ticks
        buckets = np.asarray(
            [
                estimator.state_id(int(spread), float(imb)) % estimator.n_imbalance_buckets + 1
                for spread, imb in zip(data.spread_ticks[:-1], data.imbalance[:-1], strict=True)
            ],
            dtype=np.int64,
        )
        for bucket in range(1, estimator.n_imbalance_buckets + 1):
            mask = buckets == bucket
            local_bucket_mask = mask & local_mask
            if not np.any(mask):
                continue
            rows.append(
                {
                    "split": split,
                    "imbalance_bucket": int(bucket),
                    "n": int(np.sum(mask)),
                    "local_n": int(np.sum(local_bucket_mask)),
                    "mean_imbalance": float(np.mean(data.imbalance[:-1][mask])),
                    "mean_adjustment_ticks": float(np.mean(adjustment[mask])),
                    "mean_next_mid_move_ticks": float(np.mean(next_mid_move[mask])),
                    "mean_local_next_mid_move_ticks": float(np.mean(next_mid_move[local_bucket_mask]))
                    if np.any(local_bucket_mask)
                    else 0.0,
                    "local_transition_share": float(np.mean(local_mask[mask])),
                }
            )

    df = pl.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output)
    from cmf_backtester.reporting.plots import plot_microprice_signal_diagnostics

    plot_microprice_signal_diagnostics(df, "reports/figures/microprice_signal_by_imbalance.png")
    print(df)
    print(f"Wrote {output}")
    print("Wrote reports/figures/microprice_signal_by_imbalance.png")


def cmd_diagnose_strategy_similarity(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tick_size = float(config["data"]["tick_size"])
    mid_path = Path(args.mid_timeseries)
    micro_path = Path(args.micro_timeseries)
    mid = pl.read_parquet(mid_path)
    micro = pl.read_parquet(micro_path)
    if mid.height != micro.height:
        raise ValueError("Timeseries lengths do not match")

    bid_diff_ticks = (micro["bid_quote"].to_numpy() - mid["bid_quote"].to_numpy()) / tick_size
    ask_diff_ticks = (micro["ask_quote"].to_numpy() - mid["ask_quote"].to_numpy()) / tick_size
    micro_adj_ticks = (micro["microprice"].to_numpy() - micro["mid_price"].to_numpy()) / tick_size
    quoted_spread_ticks = (mid["ask_quote"].to_numpy() - mid["bid_quote"].to_numpy()) / tick_size
    pnl_diff = micro["pnl"].to_numpy() - mid["pnl"].to_numpy()
    inventory_diff = micro["inventory"].to_numpy() - mid["inventory"].to_numpy()
    finite_quotes = np.isfinite(bid_diff_ticks) & np.isfinite(ask_diff_ticks)
    finite_spread = quoted_spread_ticks[np.isfinite(quoted_spread_ticks)]
    abs_adj = np.abs(micro_adj_ticks[np.isfinite(micro_adj_ticks)])
    abs_bid_diff = np.abs(bid_diff_ticks[finite_quotes])
    abs_ask_diff = np.abs(ask_diff_ticks[finite_quotes])

    row = {
        "rows": int(mid.height),
        "same_bid_share": float(np.mean(abs_bid_diff <= 1e-9)),
        "same_ask_share": float(np.mean(abs_ask_diff <= 1e-9)),
        "same_both_quotes_share": float(
            np.mean((abs_bid_diff <= 1e-9) & (abs_ask_diff <= 1e-9))
        ),
        "median_abs_bid_diff_ticks": float(np.quantile(abs_bid_diff, 0.5)),
        "p99_abs_bid_diff_ticks": float(np.quantile(abs_bid_diff, 0.99)),
        "median_abs_ask_diff_ticks": float(np.quantile(abs_ask_diff, 0.5)),
        "p99_abs_ask_diff_ticks": float(np.quantile(abs_ask_diff, 0.99)),
        "median_abs_microprice_adjustment_ticks": float(np.quantile(abs_adj, 0.5)),
        "p99_abs_microprice_adjustment_ticks": float(np.quantile(abs_adj, 0.99)),
        "max_abs_microprice_adjustment_ticks": float(np.max(abs_adj)),
        "median_quoted_spread_ticks": float(np.quantile(finite_spread, 0.5)),
        "avg_quoted_spread_ticks": float(np.mean(finite_spread)),
        "p99_adjustment_to_spread_ratio": float(
            np.quantile(abs_adj, 0.99) / max(float(np.mean(finite_spread)), 1e-12)
        ),
        "final_pnl_diff_micro_minus_mid": float(pnl_diff[-1]),
        "max_abs_pnl_diff": float(np.max(np.abs(pnl_diff))),
        "fill_count_diff_micro_minus_mid": int(micro["fill_count"][-1] - mid["fill_count"][-1]),
        "inventory_equal_share": float(np.mean(np.abs(inventory_diff) <= 1e-12)),
        "final_inventory_diff_micro_minus_mid": float(
            micro["inventory"][-1] - mid["inventory"][-1]
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(output)
    print(pl.DataFrame([row]))
    print(f"Wrote {output}")


def _update_final_performance(metrics: dict[str, Any]) -> None:
    path = Path("reports/tables/final_performance.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in metrics.items()}
    if path.exists():
        old = pl.read_csv(path)
        if "strategy" in old.columns and "split" in old.columns:
            old = old.filter(
                ~(
                    (pl.col("strategy") == str(row.get("strategy")))
                    & (pl.col("split") == str(row.get("split")))
                )
            )
        df = pl.concat([old, pl.DataFrame([row])], how="diagonal_relaxed")
    else:
        df = pl.DataFrame([row])
    df.write_csv(path)


def cmd_run_backtest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result, metrics = _run_backtest_from_config(config)
    split = str(metrics["split"])
    strategy_name = str(metrics["strategy"])
    result_path = Path("reports/tables") / f"{strategy_name}_{split}_timeseries.parquet"
    result.write_parquet(result_path)
    metrics_path = Path("reports/tables") / f"{strategy_name}_{split}_metrics.csv"
    pl.DataFrame([metrics]).write_csv(metrics_path)
    _update_final_performance(metrics)
    from cmf_backtester.reporting.plots import plot_quotes

    plot_quotes(result, Path("reports/figures") / f"{strategy_name}_{split}_quotes.png")
    print(f"Wrote {result_path}")
    print(f"Wrote {metrics_path}")
    print(metrics)


def cmd_calibrate(args: argparse.Namespace) -> None:
    grid_config = load_config(args.config)
    base_config = load_config(grid_config["base_config"])
    arrays = _load_arrays_from_config(base_config)
    objective = grid_config.get("objective", {})
    df = run_validation_grid(
        base_config,
        arrays,
        grid_config["grid"],
        "reports/tables/calibration_results.csv",
        float(objective.get("inventory_penalty", 0.0)),
        float(objective.get("drawdown_penalty", 0.0)),
    )
    print(df.head(10))


def _parse_csv_numbers(raw: str, value_type: type = float) -> list:
    return [value_type(x.strip()) for x in raw.split(",") if x.strip()]


def cmd_estimate_k(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    arrays = _load_arrays_from_config(config)
    train = arrays.subset(split_mask(arrays.split, "train"))
    distances = _parse_csv_numbers(args.distances, int)
    horizons = _parse_csv_numbers(args.horizons, float)
    diffs = np.diff(train.timestamps.astype(np.int64)).astype(np.float64) / 1_000_000.0
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size == 0:
        raise ValueError("Cannot estimate k: train timestamps do not have positive differences")
    median_dt_seconds = float(np.median(positive_diffs))

    rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    for horizon_seconds in horizons:
        horizon_events = max(1, int(round(horizon_seconds / median_dt_seconds)))
        probs = estimate_crossing_probabilities(
            train.best_bid_ticks,
            train.best_ask_ticks,
            distances,
            horizon_events,
        )
        distance_arr = np.asarray(distances, dtype=np.float64)
        prob_arr = np.asarray([probs[d] for d in distances], dtype=np.float64)
        fit_mask = (prob_arr > 1e-4) & (prob_arr < 0.95)
        if int(np.sum(fit_mask)) >= 2:
            k_fit = fit_exponential_k(distance_arr[fit_mask], prob_arr[fit_mask], horizon_seconds)
            n_fit_points = int(np.sum(fit_mask))
        else:
            k_fit = fit_exponential_k(distance_arr, prob_arr, horizon_seconds)
            n_fit_points = int(len(distances))
        summary_rows.append(
            {
                "horizon_seconds": float(horizon_seconds),
                "horizon_events": int(horizon_events),
                "median_dt_seconds": median_dt_seconds,
                "k_fit": float(k_fit),
                "n_fit_points": int(n_fit_points),
            }
        )
        for distance in distances:
            probability = float(probs[distance])
            intensity = float(-np.log(max(1e-12, 1.0 - probability)) / max(horizon_seconds, 1e-12))
            rows.append(
                {
                    "horizon_seconds": float(horizon_seconds),
                    "horizon_events": int(horizon_events),
                    "distance_ticks": int(distance),
                    "crossing_probability": probability,
                    "intensity_per_second": intensity,
                    "k_fit": float(k_fit),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output)
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(summary_output)
    print(summary)
    print(f"Wrote {output}")
    print(f"Wrote {summary_output}")


def cmd_run_sensitivity(args: argparse.Namespace) -> None:
    cmd_calibrate(args)
    from cmf_backtester.reporting.plots import plot_validation_scores

    results_path = Path("reports/tables/calibration_results.csv")
    if results_path.exists():
        plot_validation_scores(
            pl.read_csv(results_path),
            "reports/figures/validation_score_ranking.png",
        )
        print("Wrote reports/figures/validation_score_ranking.png")


def cmd_run_historical_experiments(args: argparse.Namespace) -> None:
    mid_config = load_config(args.mid_config)
    micro_config = load_config(args.micro_config)
    estimator_path = Path(micro_config["microprice"]["estimator_path"])
    if not estimator_path.exists():
        print("Microprice estimator not found; fitting it first.")
        namespace = argparse.Namespace(config=args.micro_config)
        cmd_fit_microprice(namespace)

    results = []
    metrics_rows = []
    by_date_rows = []
    for config in [mid_config, micro_config]:
        result, metrics = _run_backtest_from_config(config)
        results.append(result)
        metrics_rows.append(metrics)
        arrays = _load_arrays_from_config(config)
        arrays = arrays.subset(split_mask(arrays.split, str(metrics["split"])))
        by_date_rows.extend(summarize_performance_by_date(result, arrays.date))
        split = str(metrics["split"])
        strategy_name = str(metrics["strategy"])
        result.write_parquet(Path("reports/tables") / f"{strategy_name}_{split}_timeseries.parquet")
        pl.DataFrame([metrics]).write_csv(
            Path("reports/tables") / f"{strategy_name}_{split}_metrics.csv"
        )
        _update_final_performance(metrics)

    comparison_path = Path("reports/tables/historical_experiment_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(metrics_rows).write_csv(comparison_path)
    by_date_path = Path("reports/tables/historical_experiment_by_date.csv")
    pl.DataFrame(by_date_rows).write_csv(by_date_path)

    from cmf_backtester.reporting.plots import plot_inventory, plot_pnl, plot_quotes

    plot_pnl(results, "reports/figures/historical_pnl_comparison.png")
    plot_inventory(results, "reports/figures/historical_inventory_comparison.png")
    for result in results:
        plot_quotes(result, Path("reports/figures") / f"{result.strategy_name}_test_quotes.png")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {by_date_path}")
    print("Wrote historical comparison figures")


def cmd_run_monte_carlo(args: argparse.Namespace) -> None:
    from cmf_backtester.experiments.monte_carlo import (
        MonteCarloConfig,
        write_monte_carlo_outputs,
    )
    from cmf_backtester.reporting.plots import plot_monte_carlo_pnl

    config = load_config(args.config)
    raw = config.get("monte_carlo", {})
    mc_cfg = MonteCarloConfig(
        n_paths=int(raw.get("n_paths", 1000)),
        n_steps=int(raw.get("n_steps", 1000)),
        dt=float(raw.get("dt", 0.01)),
        initial_mid=float(raw.get("initial_mid", 100.0)),
        sigma=float(raw.get("sigma", 1.0)),
        gamma=float(raw.get("gamma", 0.1)),
        k=float(raw.get("k", 1.5)),
        a=float(raw.get("a", 1.0)),
        tau=float(raw.get("tau", 1.0)),
        order_size=float(raw.get("order_size", 1.0)),
        seed=int(raw.get("seed", 7)),
    )
    results, summary = write_monte_carlo_outputs(
        mc_cfg,
        "reports/tables/monte_carlo_paths.csv",
        "reports/tables/monte_carlo_summary.csv",
    )
    plot_monte_carlo_pnl(results, "reports/figures/monte_carlo_pnl_distribution.png")
    print(summary)
    print("Wrote reports/tables/monte_carlo_paths.csv")
    print("Wrote reports/tables/monte_carlo_summary.csv")
    print("Wrote reports/figures/monte_carlo_pnl_distribution.png")


def cmd_make_report(args: argparse.Namespace) -> None:
    from cmf_backtester.reporting.report import write_performance_report

    data_audit: dict[str, Any] | None = None
    audit_path = Path("reports/tables/data_audit.csv")
    if audit_path.exists():
        data_audit = pl.read_csv(audit_path).row(0, named=True)
    figures = sorted(
        path for path in Path("reports/figures").glob("*.png") if not path.name.startswith("test_")
    )
    write_performance_report(
        "reports/performance_report.md",
        data_audit=data_audit,
        metrics_table_path="reports/tables/final_performance.csv",
        figure_paths=figures,
    )
    print("Wrote reports/performance_report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cmf-backtester")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess-lob")
    p.add_argument("--config", default="configs/as_mid.yaml")
    p.add_argument("--sample-rows", type=int, default=10_000)
    p.set_defaults(func=cmd_preprocess_lob)

    p = sub.add_parser("audit-data")
    p.add_argument("--config", default="configs/as_mid.yaml")
    p.set_defaults(func=cmd_audit_data)

    p = sub.add_parser("fit-microprice")
    p.add_argument("--config", default="configs/as_microprice.yaml")
    p.set_defaults(func=cmd_fit_microprice)

    p = sub.add_parser("run-backtest")
    p.add_argument("--config", default="configs/as_mid.yaml")
    p.set_defaults(func=cmd_run_backtest)

    p = sub.add_parser("calibrate")
    p.add_argument("--config", default="configs/validation_grid.yaml")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("estimate-k")
    p.add_argument("--config", default="configs/as_mid.yaml")
    p.add_argument("--distances", default="0,1,2,3,5,8,13,21,34,55,89")
    p.add_argument("--horizons", default="1,2.5,5,10,30")
    p.add_argument("--output", default="reports/tables/k_estimation.csv")
    p.add_argument("--summary-output", default="reports/tables/k_estimation_summary.csv")
    p.set_defaults(func=cmd_estimate_k)

    p = sub.add_parser("run-sensitivity")
    p.add_argument("--config", default="configs/validation_grid.yaml")
    p.set_defaults(func=cmd_run_sensitivity)

    p = sub.add_parser("run-microprice-move-sensitivity")
    p.add_argument("--config", default="configs/as_microprice.yaml")
    p.add_argument("--max-mid-moves", default="1.0,2.0,5.0,10.0")
    p.add_argument("--output", default="reports/tables/microprice_move_sensitivity.csv")
    p.add_argument("--estimator-dir", default="data/processed/microprice_sensitivity")
    p.add_argument("--inventory-penalty", type=float, default=0.0)
    p.add_argument("--drawdown-penalty", type=float, default=0.0)
    p.set_defaults(func=cmd_run_microprice_move_sensitivity)

    p = sub.add_parser("run-quote-refresh-sensitivity")
    p.add_argument("--mid-config", default="configs/as_mid.yaml")
    p.add_argument("--micro-config", default="configs/as_microprice.yaml")
    p.add_argument("--refresh-seconds", default="0.1,0.25,0.5,1.0,2.0,5.0")
    p.add_argument("--output", default="reports/tables/quote_refresh_sensitivity.csv")
    p.add_argument("--inventory-penalty", type=float, default=0.0)
    p.add_argument("--drawdown-penalty", type=float, default=0.0)
    p.set_defaults(func=cmd_run_quote_refresh_sensitivity)

    p = sub.add_parser("run-volatility-sensitivity")
    p.add_argument("--mid-config", default="configs/as_mid.yaml")
    p.add_argument("--micro-config", default="configs/as_microprice.yaml")
    p.add_argument("--window-seconds", default="60,180,300,600")
    p.add_argument("--floors", default="0.05,0.1,0.2")
    p.add_argument("--multipliers", default="0.5,1.0,2.0")
    p.add_argument("--output", default="reports/tables/volatility_sensitivity.csv")
    p.add_argument("--inventory-penalty", type=float, default=0.0000001)
    p.add_argument("--drawdown-penalty", type=float, default=0.1)
    p.set_defaults(func=cmd_run_volatility_sensitivity)

    p = sub.add_parser("diagnose-microprice-signal")
    p.add_argument("--config", default="configs/as_microprice.yaml")
    p.add_argument("--output", default="reports/tables/microprice_signal_diagnostics.csv")
    p.set_defaults(func=cmd_diagnose_microprice_signal)

    p = sub.add_parser("diagnose-strategy-similarity")
    p.add_argument("--config", default="configs/as_microprice.yaml")
    p.add_argument(
        "--mid-timeseries",
        default="reports/tables/avellaneda_stoikov_mid_test_timeseries.parquet",
    )
    p.add_argument(
        "--micro-timeseries",
        default="reports/tables/avellaneda_stoikov_microprice_test_timeseries.parquet",
    )
    p.add_argument("--output", default="reports/tables/strategy_similarity_diagnostics.csv")
    p.set_defaults(func=cmd_diagnose_strategy_similarity)

    p = sub.add_parser("run-historical-experiments")
    p.add_argument("--mid-config", default="configs/as_mid.yaml")
    p.add_argument("--micro-config", default="configs/as_microprice.yaml")
    p.set_defaults(func=cmd_run_historical_experiments)

    p = sub.add_parser("run-monte-carlo")
    p.add_argument("--config", default="configs/monte_carlo.yaml")
    p.set_defaults(func=cmd_run_monte_carlo)

    p = sub.add_parser("make-report")
    p.set_defaults(func=cmd_make_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
