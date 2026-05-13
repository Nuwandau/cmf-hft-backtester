from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def _format_value(value: Any) -> str:
    def trim(raw: str) -> str:
        return raw.rstrip("0").rstrip(".") if "." in raw else raw

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return trim(f"{value:,.2f}")
        if abs(value) >= 10:
            return trim(f"{value:.3f}")
        if abs(value) >= 1:
            return trim(f"{value:.4f}")
        if value == 0:
            return "0"
        return trim(f"{value:.6f}")
    return str(value)


def _polars_to_markdown(df: pl.DataFrame) -> str:
    if df.height == 0:
        return "_No rows._"
    cols = df.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.iter_rows(named=True):
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _table_from_csv(
    path: str | Path,
    columns: list[str],
    *,
    head: int | None = None,
    sort_by: str | list[str] | None = None,
    descending: bool | list[bool] = False,
    filter_expr: pl.Expr | None = None,
    rename: dict[str, str] | None = None,
) -> pl.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if filter_expr is not None:
        df = df.filter(filter_expr)
    available = [col for col in columns if col in df.columns]
    if not available:
        return None
    df = df.select(available)
    if sort_by is not None:
        df = df.sort(sort_by, descending=descending)
    if head is not None:
        df = df.head(head)
    if rename:
        df = df.rename({k: v for k, v in rename.items() if k in df.columns})
    return df


def _add_table(
    lines: list[str],
    title: str,
    path: str | Path,
    columns: list[str],
    *,
    note: str | None = None,
    head: int | None = None,
    sort_by: str | list[str] | None = None,
    descending: bool | list[bool] = False,
    filter_expr: pl.Expr | None = None,
    rename: dict[str, str] | None = None,
) -> None:
    df = _table_from_csv(
        path,
        columns,
        head=head,
        sort_by=sort_by,
        descending=descending,
        filter_expr=filter_expr,
        rename=rename,
    )
    if df is None:
        return
    lines.append(f"## {title}")
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines.append(_polars_to_markdown(df))
    lines.append("")
    lines.append(f"Full table: `{Path(path).as_posix()}`")
    lines.append("")


def write_performance_report(
    output_path: str | Path,
    data_audit: dict[str, Any] | None = None,
    metrics_table_path: str | Path | None = None,
    figure_paths: list[str | Path] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.extend(
        [
            "# HFT Market-Making Backtest Report",
            "",
            "## Objective",
            "",
            "Develop an event-driven LOB replay backtester and compare Avellaneda-Stoikov "
            "market-making with mid-price and finite-state microprice references.",
            "",
            "## Current Experiment Setup",
            "",
            "- Historical replay uses L1 top-of-book features extracted from 25-level L2 snapshots.",
            "- Execution uses crossing rules: buy fills when future best ask crosses our bid, "
            "sell fills when future best bid crosses our ask.",
            "- Orders placed at timestamp `t` are eligible only from later snapshots.",
            "- Final historical runs use `visible_size` partial-fill approximation, zero fees, "
            "no latency, and no queue-position model.",
            "- The final AS configs use `quote_refresh_seconds = 0.25` and "
            "`microprice.max_mid_move_ticks = 10.0`, both selected or checked on validation.",
            "",
            "## Strategy Models",
            "",
            "```text",
            "r_t = reference_t - q_t * gamma * sigma_t^2 * tau",
            "spread_t = gamma * sigma_t^2 * tau + 2/gamma * log(1 + gamma/k)",
            "bid_t = r_t - spread_t / 2",
            "ask_t = r_t + spread_t / 2",
            "```",
            "",
            "`reference_t` is the mid-price in the baseline strategy and the finite-state "
            "microprice in the enhanced strategy. Portfolio inventory is stored in raw "
            "amount units; the AS inventory term uses `portfolio_inventory / inventory_risk_unit`.",
            "",
        ]
    )

    if data_audit:
        lines.append("## Data Audit")
        lines.append("")
        keys = [
            "rows",
            "timestamp_order_violations",
            "duplicate_timestamps",
            "median_spread_ticks",
            "p99_spread_ticks",
            "max_spread_ticks",
            "mean_imbalance",
        ]
        compact = {key: data_audit[key] for key in keys if key in data_audit}
        for key, value in compact.items():
            lines.append(f"- `{key}`: `{_format_value(value)}`")
        lines.append("")

    _add_table(
        lines,
        "Data Regime By Date",
        "reports/tables/data_audit_by_date.csv",
        [
            "date",
            "split",
            "rows",
            "median_spread_ticks",
            "p99_spread_ticks",
            "fraction_one_tick_spread",
            "fraction_spread_gt_10_ticks",
            "mean_imbalance",
        ],
    )

    if metrics_table_path:
        _add_table(
            lines,
            "Final Historical Performance",
            metrics_table_path,
            [
                "strategy",
                "final_pnl",
                "max_drawdown",
                "turnover",
                "fill_count",
                "final_inventory",
                "max_abs_inventory",
                "avg_quoted_spread_ticks",
            ],
            note=(
                "Final performance is out-of-sample on the test split. PnL is gross "
                "mark-to-market PnL under crossing execution."
            ),
        )

    _add_table(
        lines,
        "Validation Grid: AS Parameters",
        "reports/tables/calibration_results.csv",
        [
            "gamma",
            "k",
            "tau_seconds",
            "max_inventory",
            "quote_refresh_seconds",
            "score",
            "final_pnl",
        ],
        head=12,
        sort_by="score",
        descending=True,
        note=(
            "Grid search covers `gamma`, `k`, `tau`, inventory limit, and quote refresh. "
            "The table shows the top validation configurations; the full CSV keeps every run."
        ),
    )

    _add_table(
        lines,
        "Volatility Sensitivity: Mid-Price AS",
        "reports/tables/volatility_sensitivity.csv",
        [
            "vol_window_seconds",
            "vol_floor",
            "sigma_multiplier",
            "score",
            "final_pnl",
            "fill_count",
            "avg_quoted_spread_ticks",
        ],
        head=6,
        filter_expr=(pl.col("split") == "validation")
        & (pl.col("strategy") == "avellaneda_stoikov_mid"),
        sort_by="score",
        descending=True,
        note=(
            "This check varies the realized-volatility window, volatility floor, and a "
            "direct multiplier on sigma. It tests sensitivity to the `sigma_t` input in AS; "
            "the full CSV contains both validation and test rows."
        ),
    )
    _add_table(
        lines,
        "Volatility Sensitivity: Microprice AS",
        "reports/tables/volatility_sensitivity.csv",
        [
            "vol_window_seconds",
            "vol_floor",
            "sigma_multiplier",
            "score",
            "final_pnl",
            "fill_count",
            "avg_quoted_spread_ticks",
        ],
        head=6,
        filter_expr=(pl.col("split") == "validation")
        & (pl.col("strategy") == "avellaneda_stoikov_microprice"),
        sort_by="score",
        descending=True,
    )

    _add_table(
        lines,
        "Empirical k Diagnostic",
        "reports/tables/k_estimation_summary.csv",
        ["horizon_seconds", "horizon_events", "median_dt_seconds", "k_fit", "n_fit_points"],
        note=(
            "`k` is estimated on train from hypothetical quote crossing probabilities. "
            "It is used to choose a plausible validation grid, not as an exact fill model."
        ),
    )

    _add_table(
        lines,
        "Microprice Move-Filter Sensitivity",
        "reports/tables/microprice_move_sensitivity.csv",
        [
            "max_mid_move_ticks",
            "split",
            "score",
            "final_pnl",
            "fill_count",
            "filtered_share",
            "max_abs_adjustment_ticks",
        ],
        sort_by=["split", "score"],
        descending=[False, True],
    )

    _add_table(
        lines,
        "Quote Refresh Sensitivity",
        "reports/tables/quote_refresh_sensitivity.csv",
        [
            "strategy",
            "quote_refresh_seconds",
            "split",
            "score",
            "final_pnl",
            "fill_count",
            "avg_abs_inventory",
        ],
        head=24,
        sort_by=["strategy", "split", "score"],
        descending=[False, False, True],
    )

    _add_table(
        lines,
        "Microprice Signal Diagnostics",
        "reports/tables/microprice_signal_diagnostics.csv",
        [
            "imbalance_bucket",
            "n",
            "mean_imbalance",
            "mean_adjustment_ticks",
            "mean_local_next_mid_move_ticks",
            "local_transition_share",
        ],
        filter_expr=pl.col("split") == "train",
        note=(
            "A correct directional microprice should be negative at low bid imbalance "
            "and positive at high bid imbalance."
        ),
    )

    _add_table(
        lines,
        "Strategy Similarity Diagnostics",
        "reports/tables/strategy_similarity_diagnostics.csv",
        [
            "same_both_quotes_share",
            "median_abs_microprice_adjustment_ticks",
            "p99_abs_microprice_adjustment_ticks",
            "avg_quoted_spread_ticks",
            "p99_adjustment_to_spread_ratio",
            "final_pnl_diff_micro_minus_mid",
            "fill_count_diff_micro_minus_mid",
            "inventory_equal_share",
        ],
        note=(
            "This explains why mid-price and microprice AS can remain close in PnL: "
            "the fitted microprice adjustment is still small relative to the AS quoted "
            "spread, even when rounded quotes differ frequently."
        ),
    )

    _add_table(
        lines,
        "Historical Test Contribution By Date",
        "reports/tables/historical_experiment_by_date.csv",
        [
            "strategy",
            "date",
            "pnl_contribution",
            "turnover_contribution",
            "fill_count",
            "end_inventory",
            "max_abs_inventory",
            "avg_quoted_spread_ticks",
        ],
    )

    _add_table(
        lines,
        "Monte Carlo Simulation",
        "reports/tables/monte_carlo_summary.csv",
        [
            "strategy",
            "mean_pnl",
            "std_pnl",
            "p05_pnl",
            "p95_pnl",
            "mean_final_inventory",
            "mean_abs_inventory",
            "mean_turnover",
            "mean_fill_count",
        ],
    )

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for path in figure_paths:
            path = Path(path)
            if path.exists():
                lines.append(f"![{path.stem}]({path.as_posix()})")
                lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- `visible_size` partial fills cap quantity by top-of-book displayed size, but this is not a queue-position model.",
            "- No latency model.",
            "- No transaction fees or rebates in the final baseline.",
            "- Trades are not used for baseline fills because the `side` convention is not independently documented.",
            "- No forced terminal liquidation; final inventory may be non-zero.",
            "",
            "## Improvement Roadmap",
            "",
            "- Queue-position partial fills.",
            "- Trade-based execution after stronger feed synchronization checks.",
            "- More robust walk-forward calibration of `k`, volatility, and risk parameters.",
            "- Rolling microprice recalibration and L2 depth features.",
            "- Fees, rebates, latency, and forced liquidation.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
