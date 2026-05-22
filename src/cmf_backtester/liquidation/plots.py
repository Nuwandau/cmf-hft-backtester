from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cmf_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _group_value(value):
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _signed_bucket_order(label: str) -> int:
    if label == "zero":
        return 100
    prefix, _, power = label.partition("_1e")
    try:
        exponent = int(power)
    except ValueError:
        return 100
    if prefix == "neg":
        return 90 - exponent
    if prefix == "pos":
        return 110 + exponent
    return 100


def _save(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_event_counts_by_day(daily: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for (source, symbol), part in daily.group_by(["source", "symbol"], maintain_order=True):
        pdf = part.sort("date")
        x_values = np.array(pdf["date"].to_list(), dtype="datetime64[D]")
        ax.plot(x_values, pdf["rows"].to_list(), label=f"{source}:{symbol}", linewidth=1)
    ax.set_title("Events By Day")
    ax.set_xlabel("date")
    ax.set_ylabel("rows, log scale")
    ax.set_yscale("log")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, output_path)


def plot_hist(
    sample: pl.DataFrame,
    column: str,
    output_path: str | Path,
    title: str,
    bins: int = 80,
    *,
    xlabel: str | None = None,
    log_y: bool = False,
    clip_quantile: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    values = sample[column].drop_nulls().to_numpy()
    values = values[np.isfinite(values)]
    if clip_quantile is not None and values.size:
        upper = np.quantile(values, clip_quantile)
        values = values[values <= upper]
    if values.size:
        ax.hist(values, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("count")
    if log_y:
        ax.set_yscale("log")
    _save(fig, output_path)


def plot_queue_imbalance(table: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ["split", "symbol"] if "split" in table.columns else ["symbol"]
    for values, part in table.group_by(groups, maintain_order=True):
        if not isinstance(values, tuple):
            values = (values,)
        label = ":".join(str(_group_value(value)) for value in values)
        ax.plot(
            part.sort("imbalance_bucket")["imbalance_bucket"],
            part.sort("imbalance_bucket")["prob_next_move_up"],
            marker="o",
            label=f"{label} up",
        )
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.set_title("Conditional Next Mid Move Direction By Queue Imbalance")
    ax.set_xlabel("queue imbalance bucket")
    ax.set_ylabel("P(next mid up | mid moved)")
    ax.legend()
    _save(fig, output_path)


def plot_ofi_response(table: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if table.height and "corr_ofi_next_return_bps" in table.columns:
        labels = (
            [f"{row['split']}:{row['symbol']}" for row in table.select(["split", "symbol"]).to_dicts()]
            if "split" in table.columns
            else table["symbol"].to_list()
        )
        ax.bar(labels, table["corr_ofi_next_return_bps"].fill_null(0).to_list())
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("OFI Correlation With Next Sample Return")
    ax.set_xlabel("symbol")
    ax.set_ylabel("correlation")
    ax.tick_params(axis="x", labelrotation=25)
    _save(fig, output_path)


def plot_markout_distribution(markouts: pl.DataFrame, horizon: int, output_path: str | Path) -> None:
    col = f"pnl_bps_{horizon}s"
    fig, ax = plt.subplots(figsize=(10, 5))
    for (symbol, side), part in markouts.group_by(["symbol", "side"], maintain_order=True):
        values = part[col].drop_nulls().to_numpy()
        if values.size:
            ax.hist(values, bins=80, alpha=0.35, density=True, label=f"{symbol}:{side}")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Maker Markout Distribution, {horizon}s")
    ax.set_xlabel("PnL bps")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_markout_curve(summary: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.height:
        groups = ["split", "symbol", "side"] if "split" in summary.columns else ["symbol", "side"]
        for values, part in summary.group_by(groups, maintain_order=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = ":".join(str(_group_value(value)) for value in values)
            ordered = part.sort("horizon_seconds")
            ax.plot(
                ordered["horizon_seconds"],
                ordered["weighted_pnl_bps"],
                marker="o",
                label=label,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Weighted Maker Markout Curve")
    ax.set_xlabel("horizon, seconds")
    ax.set_ylabel("weighted PnL bps")
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_response_functions(table: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if table.height:
        groups = (
            ["split", "symbol", "flow_type"]
            if "split" in table.columns
            else ["symbol", "flow_type"]
        )
        for values, part in table.group_by(groups, maintain_order=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = ":".join(str(_group_value(value)) for value in values)
            ordered = part.sort("horizon_seconds")
            ax.plot(
                ordered["horizon_seconds"],
                ordered["response_bps"] if "response_bps" in ordered.columns else ordered["response_bps_per_musd"],
                marker="o",
                label=label,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Signed-Flow Response Functions")
    ax.set_xlabel("horizon, seconds")
    ax.set_ylabel("signed-flow weighted response, bps")
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nonlinear_response(table: pl.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if table.height:
        bucket_col = "signed_liq_bucket" if "signed_liq_bucket" in table.columns else "signed_liq_quantile"
        plot_table = table
        if "window_seconds" in plot_table.columns:
            plot_table = plot_table.filter(pl.col("window_seconds") == 30)
        if "horizon_seconds" in plot_table.columns:
            plot_table = plot_table.filter(pl.col("horizon_seconds") == 30)
        if {"split", "venue", "clipped_turnover", "weighted_pnl_bps"}.issubset(set(plot_table.columns)):
            plot_table = plot_table.group_by(["split", "venue", bucket_col]).agg(
                (
                    (pl.col("weighted_pnl_bps") * pl.col("clipped_turnover")).sum()
                    / pl.col("clipped_turnover").sum()
                ).alias("weighted_pnl_bps")
            )
            groups = ["split", "venue"]
        else:
            groups = ["symbol"]
        for values, part in plot_table.group_by(groups, maintain_order=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = ":".join(str(_group_value(value)) for value in values)
            rows = sorted(part.to_dicts(), key=lambda row: _signed_bucket_order(str(row[bucket_col])))
            ax.plot(
                [str(row[bucket_col]) for row in rows],
                [row["weighted_pnl_bps"] for row in rows],
                marker="o",
                label=label,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Markout By Signed Liquidation Pressure Bucket, 30s Window / 30s Horizon")
    ax.set_xlabel("signed liquidation pressure bucket")
    ax.set_ylabel("weighted PnL bps")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_event_study(table: pl.DataFrame, output_path: str | Path, *, detailed: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if table.height:
        groups = ["symbol", "venue", "side"] if detailed else ["venue", "side"]
        grouped = (
            table.group_by([*groups, "offset_seconds"])
            .agg(
                (
                    (pl.col("mean_return_bps") * pl.col("rows")).sum()
                    / pl.col("rows").sum()
                ).alias("weighted_mean_return_bps")
            )
        )
        for values, part in grouped.group_by(groups, maintain_order=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = ":".join(str(_group_value(value)) for value in values)
            ordered = part.sort("offset_seconds")
            ax.plot(
                ordered["offset_seconds"],
                ordered["weighted_mean_return_bps"],
                marker="o",
                label=label,
            )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    title_suffix = "Detailed" if detailed else "Aggregated"
    ax.set_title(f"Liquidation Event Study: Binance Mid Return Around Event ({title_suffix})")
    ax.set_xlabel("offset from liquidation availability, seconds")
    ax.set_ylabel("mid return vs event time, bps")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, output_path)
