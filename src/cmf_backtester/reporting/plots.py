from __future__ import annotations

import os
from pathlib import Path

Path("/private/tmp/cmf_matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cmf_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cmf_backtester.backtest.recorder import BacktestResult
from cmf_backtester.market.microprice import MicropriceEstimator

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


def _block_mean(values, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    block = max(1, int(np.ceil(n / max_points)))
    usable = (n // block) * block
    if usable == 0:
        return np.arange(n), arr
    x = np.arange(block // 2, usable, block)
    y = arr[:usable].reshape(-1, block).mean(axis=1)
    if usable < n:
        x = np.append(x, (usable + n - 1) // 2)
        y = np.append(y, arr[usable:].mean())
    return x, y


def plot_pnl(results: list[BacktestResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(results) >= 2:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
        ax_main, ax_diff = axes
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(10, 5))
        ax_diff = None
    for result in results:
        ax_main.plot(result.pnl, label=result.strategy_name)
    ax_main.set_title("Historical PnL")
    ax_main.set_xlabel("event")
    ax_main.set_ylabel("PnL")
    ax_main.legend()
    if ax_diff is not None:
        base = results[0]
        for result in results[1:]:
            n = min(len(base.pnl), len(result.pnl))
            ax_diff.plot(
                np.arange(n),
                result.pnl[:n] - base.pnl[:n],
                label=f"{result.strategy_name} - {base.strategy_name}",
            )
        ax_diff.axhline(0.0, color="black", linewidth=0.7)
        ax_diff.set_title("PnL Difference")
        ax_diff.set_xlabel("event")
        ax_diff.set_ylabel("delta PnL")
        ax_diff.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_inventory(results: list[BacktestResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(results) >= 2:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
        ax_line, ax_diff, ax_hist = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
        ax_line, ax_hist = axes
        ax_diff = None
    for result in results:
        x, y = _block_mean(result.inventory, max_points=1500)
        ax_line.plot(x, y, label=result.strategy_name, linewidth=1.2)
    ax_line.axhline(0.0, color="black", linewidth=0.7)
    ax_line.set_title("Inventory: Block Mean And State Distribution")
    ax_line.set_xlabel("event")
    ax_line.set_ylabel("block mean inventory")
    ax_line.legend(loc="best")

    if ax_diff is not None:
        base = results[0]
        for result in results[1:]:
            n = min(len(base.inventory), len(result.inventory))
            x, y = _block_mean(result.inventory[:n] - base.inventory[:n], max_points=1500)
            ax_diff.plot(x, y, label=f"{result.strategy_name} - {base.strategy_name}")
        ax_diff.axhline(0.0, color="black", linewidth=0.7)
        ax_diff.set_title("Inventory Difference")
        ax_diff.set_xlabel("event")
        ax_diff.set_ylabel("delta inventory")
        ax_diff.legend(loc="best")

    all_states = sorted({float(v) for result in results for v in np.unique(result.inventory)})
    x_pos = np.arange(len(all_states))
    width = 0.8 / max(len(results), 1)
    for idx, result in enumerate(results):
        states, counts = np.unique(result.inventory, return_counts=True)
        count_map = {float(s): int(c) for s, c in zip(states, counts, strict=True)}
        freq = np.asarray([count_map.get(state, 0) / max(len(result.inventory), 1) for state in all_states])
        offset = (idx - (len(results) - 1) / 2) * width
        ax_hist.bar(x_pos + offset, freq, width=width, label=result.strategy_name)
    ax_hist.set_xticks(x_pos)
    ax_hist.set_xticklabels([f"{state:g}" for state in all_states])
    ax_hist.set_xlabel("inventory state")
    ax_hist.set_ylabel("fraction")
    ax_hist.legend(loc="best")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_quotes(result: BacktestResult, output_path: str | Path, start: int = 0, n: int = 1000) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end = min(len(result.timestamps), start + n)
    x = np.arange(start, end)
    tick = result.tick_size
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [2.2, 1]})
    ax_price, ax_adj = axes
    mid = result.mid_ticks[start:end] * tick
    micro = result.microprice_ticks[start:end] * tick
    reservation = result.reservation_ticks[start:end] * tick
    bid = result.bid_quote_ticks[start:end] * tick
    ask = result.ask_quote_ticks[start:end] * tick
    ax_price.plot(x, mid, label="mid", linewidth=1.4, color="black", zorder=4)
    if np.nanmax(np.abs(micro - mid)) > 1e-12:
        ax_price.plot(
            x,
            micro,
            label="microprice",
            linewidth=1.2,
            linestyle="--",
            color="tab:purple",
            zorder=4,
        )
    ax_price.plot(
        x,
        reservation,
        label="reservation price",
        linewidth=1.2,
        linestyle="-.",
        color="tab:blue",
        zorder=3,
    )
    ax_price.scatter(x, bid, label="bid quote", s=10, alpha=0.45, color="tab:green", zorder=2)
    ax_price.scatter(x, ask, label="ask quote", s=10, alpha=0.45, color="tab:red", zorder=2)
    finite = np.concatenate(
        [
            mid[np.isfinite(mid)],
            micro[np.isfinite(micro)],
            reservation[np.isfinite(reservation)],
            bid[np.isfinite(bid)],
            ask[np.isfinite(ask)],
        ]
    )
    if finite.size:
        lo, hi = np.quantile(finite, [0.005, 0.995])
        pad = max((hi - lo) * 0.08, tick)
        ax_price.set_ylim(lo - pad, hi + pad)
    ax_price.set_title(f"Quotes Sample: {result.strategy_name}")
    ax_price.set_xlabel("event")
    ax_price.set_ylabel("price")
    ax_price.legend(ncol=3, fontsize=9)

    adjustment_ticks = result.microprice_ticks[start:end] - result.mid_ticks[start:end]
    ax_adj.plot(x, adjustment_ticks, color="tab:purple", linewidth=1.0)
    ax_adj.axhline(0.0, color="black", linewidth=0.7)
    ax_adj.set_title("Microprice Adjustment")
    ax_adj.set_xlabel("event")
    ax_adj.set_ylabel("ticks")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_microprice_adjustment(estimator: MicropriceEstimator, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    buckets = list(range(1, estimator.n_imbalance_buckets + 1))
    for spread_idx in range(min(estimator.n_spread_states, 5)):
        start = spread_idx * estimator.n_imbalance_buckets
        end = start + estimator.n_imbalance_buckets
        label = f"spread_state={spread_idx + 1}"
        plt.plot(buckets, estimator.adjustment_ticks[start:end], marker="o", label=label)
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.title("Microprice adjustment by imbalance bucket")
    plt.xlabel("imbalance bucket")
    plt.ylabel("adjustment, ticks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_monte_carlo_pnl(results: pl.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for strategy in results["strategy"].unique().to_list():
        pnl = results.filter(pl.col("strategy") == strategy)["pnl"].to_numpy()
        plt.hist(pnl, bins=50, alpha=0.55, label=strategy)
    plt.title("Monte Carlo PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_validation_scores(results: pl.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "score" not in results.columns:
        return
    top = results.sort("score", descending=True).head(30).with_row_index("rank")
    plt.figure(figsize=(12, 5))
    plt.plot(top["rank"].to_numpy(), top["score"].to_numpy(), marker="o")
    plt.title("Top Validation Scores")
    plt.xlabel("rank")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_quote_refresh_sensitivity(results: pl.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax_pnl, ax_fills = axes
    for split in ["validation", "test"]:
        for strategy in results["strategy"].unique().to_list():
            subset = (
                results.filter((pl.col("split") == split) & (pl.col("strategy") == strategy))
                .sort("quote_refresh_seconds")
            )
            if subset.height == 0:
                continue
            label = f"{strategy} / {split}"
            ax_pnl.plot(
                subset["quote_refresh_seconds"].to_numpy(),
                subset["final_pnl"].to_numpy(),
                marker="o",
                label=label,
            )
            ax_fills.plot(
                subset["quote_refresh_seconds"].to_numpy(),
                subset["fill_count"].to_numpy(),
                marker="o",
                label=label,
            )
    ax_pnl.set_title("Quote Refresh Sensitivity")
    ax_pnl.set_xlabel("quote refresh, seconds")
    ax_pnl.set_ylabel("final PnL")
    ax_pnl.legend(fontsize=8)
    ax_fills.set_xlabel("quote refresh, seconds")
    ax_fills.set_ylabel("fill count")
    ax_fills.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_microprice_signal_diagnostics(results: pl.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax_adj, ax_move = axes
    for split in ["train", "validation", "test"]:
        subset = results.filter(pl.col("split") == split).sort("imbalance_bucket")
        if subset.height == 0:
            continue
        x = subset["imbalance_bucket"].to_numpy()
        ax_adj.plot(x, subset["mean_adjustment_ticks"].to_numpy(), marker="o", label=split)
        ax_move.plot(x, subset["mean_local_next_mid_move_ticks"].to_numpy(), marker="o", label=split)
    ax_adj.axhline(0.0, color="black", linewidth=0.7)
    ax_adj.set_title("Microprice Adjustment By Imbalance")
    ax_adj.set_xlabel("imbalance bucket")
    ax_adj.set_ylabel("mean adjustment, ticks")
    ax_adj.legend()
    ax_move.axhline(0.0, color="black", linewidth=0.7)
    ax_move.set_title("Next Local Mid Move By Imbalance")
    ax_move.set_xlabel("imbalance bucket")
    ax_move.set_ylabel("mean next move, ticks")
    ax_move.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_volatility_sensitivity(results: pl.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax_pnl, ax_spread = axes
    for strategy in results["strategy"].unique().to_list():
        subset = results.filter((pl.col("split") == "validation") & (pl.col("strategy") == strategy))
        if subset.height == 0:
            continue
        grouped = (
            subset.group_by("sigma_multiplier")
            .agg(
                [
                    pl.col("final_pnl").mean().alias("mean_final_pnl"),
                    pl.col("avg_quoted_spread_ticks").mean().alias("mean_spread"),
                ]
            )
            .sort("sigma_multiplier")
        )
        ax_pnl.plot(
            grouped["sigma_multiplier"].to_numpy(),
            grouped["mean_final_pnl"].to_numpy(),
            marker="o",
            label=strategy,
        )
        ax_spread.plot(
            grouped["sigma_multiplier"].to_numpy(),
            grouped["mean_spread"].to_numpy(),
            marker="o",
            label=strategy,
        )
    ax_pnl.set_title("Volatility Sensitivity: Validation")
    ax_pnl.set_xlabel("sigma multiplier")
    ax_pnl.set_ylabel("mean final PnL")
    ax_pnl.legend(fontsize=8)
    ax_spread.set_xlabel("sigma multiplier")
    ax_spread.set_ylabel("mean quoted spread, ticks")
    ax_spread.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
