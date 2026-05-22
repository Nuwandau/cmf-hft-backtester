from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from cmf_backtester.liquidation.config import LiquidationEdaConfig, load_liquidation_config
from cmf_backtester.liquidation.eda import (
    add_split_column,
    bbo_ofi_summary,
    bbo_quality,
    convention_examples,
    daily_stability,
    liquidation_summary,
    nonlinear_flow_response,
    price_location_summary,
    queue_imbalance_next_move,
    trade_summary,
)
from cmf_backtester.liquidation.features import (
    add_bbo_features,
    add_liquidation_features,
    add_ofi,
)
from cmf_backtester.liquidation.full_eda import compute_full_data_eda
from cmf_backtester.liquidation.io import (
    collect_frame,
    daily_event_counts,
    deterministic_sample,
    ensure_output_dirs,
    hourly_event_counts,
    scan_source,
    schema_audit,
    source_file_table,
    source_quality_tables,
)
from cmf_backtester.liquidation.markout import (
    compute_markouts,
    join_trades_to_bbo,
    summarize_markouts,
)
from cmf_backtester.liquidation.plots import (
    plot_event_counts_by_day,
    plot_event_study,
    plot_hist,
    plot_markout_curve,
    plot_markout_distribution,
    plot_nonlinear_response,
    plot_ofi_response,
    plot_queue_imbalance,
    plot_response_functions,
)
from cmf_backtester.liquidation.report import write_liquidation_report
from cmf_backtester.liquidation.schema import (
    BINANCE_BBO,
    BINANCE_LIQUIDATIONS,
    BINANCE_TRADES,
    BYBIT_LIQUIDATIONS,
    SourceSpec,
    source_specs,
)


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.width == 0:
        df = pl.DataFrame({"note": ["no rows"]})
    df.write_csv(path)


def _read_existing_tables(root: Path, names: list[str]) -> dict[str, pl.DataFrame] | None:
    paths = {name: root / name for name in names}
    if not all(path.exists() for path in paths.values()):
        return None
    return {name: pl.read_csv(path) for name, path in paths.items()}


def _concat_or_empty(frames: list[pl.DataFrame]) -> pl.DataFrame:
    valid = [df for df in frames if df.height > 0 or df.width > 0]
    return pl.concat(valid, how="diagonal_relaxed") if valid else pl.DataFrame()


def _phase(metadata: dict[str, Any], name: str):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"[liquidation-eda] {name}...", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.perf_counter() - self.start
            metadata.setdefault("phase_seconds", {})[name] = elapsed
            print(f"[liquidation-eda] {name}: {elapsed:.2f}s", flush=True)

    return Timer()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _spec(specs: list[SourceSpec], source: str, symbol: str) -> SourceSpec:
    for spec in specs:
        if spec.source == source and spec.symbol == symbol:
            return spec
    raise KeyError((source, symbol))


def _prepare_bbo_sample(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    max_rows: int,
) -> pl.DataFrame:
    sample = deterministic_sample(config, spec, max_rows)
    sample = sample.with_columns(pl.lit(spec.symbol).alias("symbol"))
    sample = add_bbo_features(sample)
    sample = add_ofi(sample)
    return sample.with_columns(
        [
            ((pl.col("mid").shift(-1) - pl.col("mid")) / pl.col("mid") * 10_000).alias(
                "next_return_bps"
            ),
        ]
    )


def _prepare_trade_sample(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    max_rows: int,
) -> pl.DataFrame:
    return deterministic_sample(config, spec, max_rows).with_columns(pl.lit(spec.symbol).alias("symbol"))


def _prepare_liq_sample(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    max_rows: int,
) -> pl.DataFrame:
    venue = "bybit" if spec.source == BYBIT_LIQUIDATIONS else "binance"
    return add_liquidation_features(
        deterministic_sample(config, spec, max_rows).with_columns(pl.lit(spec.symbol).alias("symbol")),
        venue=venue,
        bybit_delay_us=config.bybit_delay_us,
    )


def _utc_us(raw: str) -> int:
    return int(datetime.fromisoformat(raw).replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def _split_bounds_us(split: str) -> tuple[int, int]:
    if split == "train":
        return _utc_us("2025-12-01T00:00:00"), _utc_us("2026-02-01T00:00:00") - 1
    if split == "validation":
        return _utc_us("2026-02-01T00:00:00"), _utc_us("2026-03-01T00:00:00") - 1
    raise ValueError(f"Unsupported split for liquidation EDA sampling: {split}")


def _sample_split_head(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    *,
    split: str,
    max_rows: int,
) -> pl.DataFrame:
    start_us, end_us = _split_bounds_us(split)
    return collect_frame(
        scan_source(config, spec)
        .filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= end_us))
        .with_row_index("original_row_id")
        .limit(max_rows)
    ).with_columns(pl.lit(spec.symbol).alias("symbol"), pl.lit(split).alias("sample_split"))


def _bbo_slice_for_events(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    events: pl.DataFrame,
) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()
    start_us = int(events["timestamp"].min()) - config.bbo_staleness_tolerance_us
    end_us = (
        int(events["timestamp"].max())
        + max(config.eda_curve_horizons_seconds) * 1_000_000
        + config.bbo_staleness_tolerance_us
    )
    bbo = collect_frame(
        scan_source(config, spec)
        .filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= end_us))
        .select(["timestamp", "ticker", "bid_price", "bid_amount", "ask_price", "ask_amount"])
    ).with_columns(pl.lit(spec.symbol).alias("symbol"))
    return add_ofi(add_bbo_features(bbo)).with_columns(
        ((pl.col("mid").shift(-1) - pl.col("mid")) / pl.col("mid") * 10_000).alias(
            "next_return_bps"
        )
    )


def _liquidation_slice_for_events(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    events: pl.DataFrame,
    *,
    venue: str,
) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()
    start_us = int(events["timestamp"].min()) - 30 * 1_000_000 - config.bybit_delay_us
    end_us = int(events["timestamp"].max()) + max(config.eda_curve_horizons_seconds) * 1_000_000
    raw = collect_frame(
        scan_source(config, spec)
        .filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= end_us))
        .select(["timestamp", "ticker", "side", "price", "amount"])
    ).with_columns(pl.lit(spec.symbol).alias("symbol"))
    return add_liquidation_features(
        raw, venue=venue, bybit_delay_us=config.bybit_delay_us
    ).with_columns(pl.lit(venue).alias("venue"))


def _add_recent_liq_pressure(
    trades: pl.DataFrame,
    liquidations: pl.DataFrame,
    *,
    timestamp_col: str,
    output_col: str,
    window_seconds: int,
) -> pl.DataFrame:
    if trades.height == 0 or liquidations.height == 0:
        return trades.with_columns(pl.lit(0.0).alias(output_col))
    liq = (
        liquidations.sort(timestamp_col)
        .with_columns(pl.col("signed_liquidation_notional").cum_sum().alias("_cum_signed_liq"))
        .select([pl.col(timestamp_col).alias("_liq_ts"), "_cum_signed_liq"])
    )
    base = trades.sort("timestamp").with_columns(
        [
            pl.col("timestamp").alias("_trade_ts"),
            (pl.col("timestamp") - window_seconds * 1_000_000).alias("_window_start_ts"),
        ]
    )
    at_trade = base.join_asof(
        liq,
        left_on="_trade_ts",
        right_on="_liq_ts",
        strategy="backward",
    ).rename({"_cum_signed_liq": "_cum_at_trade"})
    at_start = at_trade.join_asof(
        liq,
        left_on="_window_start_ts",
        right_on="_liq_ts",
        strategy="backward",
    ).rename({"_cum_signed_liq": "_cum_at_start"})
    return at_start.with_columns(
        (pl.col("_cum_at_trade").fill_null(0.0) - pl.col("_cum_at_start").fill_null(0.0)).alias(
            output_col
        )
    ).drop([col for col in ["_trade_ts", "_window_start_ts", "_liq_ts", "_cum_at_trade", "_cum_at_start"] if col in at_start.columns])


def _markout_by_liquidation_context(
    markouts: pl.DataFrame, horizons_seconds: tuple[int, ...]
) -> pl.DataFrame:
    if "signed_liq_notional_30s" not in markouts.columns:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for horizon in horizons_seconds:
        pnl_col = f"pnl_bps_{horizon}s"
        if pnl_col not in markouts.columns:
            continue
        df = markouts.filter(pl.col(pnl_col).is_not_null())
        if df.height == 0:
            continue
        df = df.with_columns(
            [
                (
                    pl.when(pl.col("signed_liq_notional_30s") > 0)
                    .then(pl.lit("upward_pressure"))
                    .when(pl.col("signed_liq_notional_30s") < 0)
                    .then(pl.lit("downward_pressure"))
                    .otherwise(pl.lit("no_sampled_pressure"))
                ).alias("liq_pressure_bucket"),
                (
                    pl.when(pl.col("maker_direction") * pl.col("signed_liq_notional_30s") > 0)
                    .then(pl.lit("same_direction_toxic_risk"))
                    .when(pl.col("maker_direction") * pl.col("signed_liq_notional_30s") < 0)
                    .then(pl.lit("opposite_direction"))
                    .otherwise(pl.lit("none"))
                ).alias("maker_vs_liq_pressure"),
            ]
        )
        grouped = (
            df.group_by(["symbol", "side", "liq_pressure_bucket", "maker_vs_liq_pressure"])
            .agg(
                pl.len().alias("rows"),
                ((pl.col(pnl_col) * pl.col("clipped_notional")).sum() / pl.col("clipped_notional").sum()).alias(
                    "weighted_pnl_bps"
                ),
                pl.col(pnl_col).median().alias("median_pnl_bps"),
                pl.col("clipped_notional").sum().alias("clipped_turnover"),
            )
            .with_columns(pl.lit(horizon).alias("horizon_seconds"))
        )
        rows.extend(grouped.to_dicts())
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _response_functions(
    events: pl.DataFrame,
    bbo: pl.DataFrame,
    *,
    symbol: str,
    flow_type: str,
    timestamp_col: str,
    signed_flow_col: str,
    horizons_seconds: tuple[int, ...],
    tolerance_us: int,
) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame()
    base_bbo = bbo.select(
        [
            pl.col("timestamp").alias("event_bbo_timestamp"),
            pl.col("mid").alias("event_mid"),
        ]
    ).sort("event_bbo_timestamp")
    future_bbo = bbo.select(
        [
            pl.col("timestamp").alias("future_bbo_timestamp"),
            pl.col("mid").alias("future_mid"),
        ]
    ).sort("future_bbo_timestamp")
    base = (
        events.select(
            [
                pl.col(timestamp_col).alias("event_timestamp"),
                pl.col(signed_flow_col).alias("signed_flow"),
            ]
        )
        .filter(pl.col("signed_flow").is_not_null())
        .sort("event_timestamp")
        .join_asof(
            base_bbo,
            left_on="event_timestamp",
            right_on="event_bbo_timestamp",
            strategy="backward",
            tolerance=tolerance_us,
        )
    )
    rows: list[dict[str, Any]] = []
    for horizon in horizons_seconds:
        target = base.with_columns(
            (pl.col("event_timestamp") + horizon * 1_000_000).alias("target_timestamp")
        )
        joined = target.sort("target_timestamp").join_asof(
            future_bbo,
            left_on="target_timestamp",
            right_on="future_bbo_timestamp",
            strategy="backward",
            tolerance=tolerance_us,
        )
        valid = joined.filter(
            pl.col("event_mid").is_not_null()
            & pl.col("future_mid").is_not_null()
            & pl.col("signed_flow").is_not_null()
        ).with_columns(((pl.col("future_mid") - pl.col("event_mid")) / pl.col("event_mid") * 10_000).alias("return_bps"))
        if valid.height == 0:
            continue
        abs_flow = float(valid["signed_flow"].abs().sum())
        response = (
            float((valid["return_bps"] * valid["signed_flow"]).sum())
            / max(abs_flow, 1e-12)
        )
        rows.append(
            {
                "symbol": symbol,
                "flow_type": flow_type,
                "horizon_seconds": horizon,
                "rows": valid.height,
                "response_bps_per_musd": response,
                "mean_return_bps": float(valid["return_bps"].mean()),
                "abs_signed_flow_musd": abs_flow / 1_000_000.0,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _anomaly_log(
    bbo_quality_df: pl.DataFrame,
    price_location_df: pl.DataFrame,
    coverage_df: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in bbo_quality_df.to_dicts():
        if row.get("crossed_rows", 0) or row.get("locked_rows", 0):
            rows.append(
                {
                    "symbol": row["symbol"],
                    "source": BINANCE_BBO,
                    "timestamp_utc": "",
                    "issue_type": "locked_or_crossed_bbo",
                    "severity": "high" if row.get("crossed_rows", 0) else "medium",
                    "notes": f"crossed={row.get('crossed_rows')}, locked={row.get('locked_rows')}",
                }
            )
    if price_location_df.height:
        odd = (
            price_location_df.filter(
                pl.col("price_location").is_in(
                    ["above_ask", "below_bid", "outside_or_ambiguous"]
                )
                & (pl.col("rows") > 0)
            )
            .group_by(["split", "symbol", "side", "price_location"])
            .agg(pl.col("rows").sum().alias("rows"))
            .sort(["symbol", "split", "side", "price_location"])
        )
        for row in odd.to_dicts():
            rows.append(
                {
                    "symbol": row["symbol"],
                    "source": BINANCE_TRADES,
                    "timestamp_utc": "",
                    "issue_type": "trade_far_or_ambiguous_vs_bbo",
                    "severity": "medium",
                    "notes": (
                        f"split={row.get('split', 'sample')} "
                        f"{row['side']} {row['price_location']} rows={row['rows']}"
                    ),
                }
            )
    for row in coverage_df.to_dicts():
        if row.get("duplicate_timestamp_rows", 0) > 0:
            rows.append(
                {
                    "symbol": row["symbol"],
                    "source": row["source"],
                    "timestamp_utc": "",
                    "issue_type": "duplicate_timestamps",
                    "severity": "low",
                    "notes": f"duplicate_timestamp_rows={row.get('duplicate_timestamp_rows')}",
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        {
            "symbol": ["all"],
            "source": ["all"],
            "timestamp_utc": [""],
            "issue_type": ["none_detected_in_sample"],
            "severity": ["info"],
            "notes": ["No anomalies triggered by implemented checks."],
        }
    )


def _write_hypotheses(config: LiquidationEdaConfig, markout_context: pl.DataFrame) -> None:
    path = Path("docs/research/notes/liquidation_signal_hypotheses.md")
    lines = [
        "# Liquidation Signal Hypotheses",
        "",
        "Generated from the liquidation EDA pipeline.",
        "",
        "## Empirical Pointers From Current EDA",
        "",
        "- Source coverage spans 2025-12-01 through 2026-02-28 for both BTCUSDT and ETHUSDT.",
        "- Binance trades have many duplicate timestamps, so same-timestamp ordering must not be inferred.",
        "- Full-data BBO quality, markout, liquidation context, response, and event-study tables are computed with daily chunks.",
        "- Deterministic samples are retained only for visual distribution plots.",
        "",
    ]
    if markout_context.height:
        best = markout_context.sort("weighted_pnl_bps", descending=True).row(0, named=True)
        worst = markout_context.sort("weighted_pnl_bps").row(0, named=True)
        lines.extend(
            [
                "- Strongest full-data liquidation-context bucket by weighted maker PnL: "
                f"{best.get('symbol')} {best.get('side')} {best.get('horizon_seconds')}s "
                f"{best.get('liq_pressure_bucket')} / {best.get('maker_vs_liq_pressure')} "
                f"= {best.get('weighted_pnl_bps'):.4f} bps.",
                "- Weakest full-data liquidation-context bucket by weighted maker PnL: "
                f"{worst.get('symbol')} {worst.get('side')} {worst.get('horizon_seconds')}s "
                f"{worst.get('liq_pressure_bucket')} / {worst.get('maker_vs_liq_pressure')} "
                f"= {worst.get('weighted_pnl_bps'):.4f} bps.",
                "",
            ]
        )
    lines.extend(
        [
        "## Strong Candidates To Test Later",
        "",
        "- Same-direction liquidation pressure may identify toxic maker trades, but the direction is not "
        "uniform across symbol/side/horizon and must be validated split-by-split.",
        "- Bybit liquidation clusters may lead Binance adverse selection after the required 200ms delay.",
        "- OFI and queue imbalance may help distinguish toxic flow from ordinary trade flow.",
        "- Extreme signed liquidation pressure may be nonlinear: saturation or reversal should be modeled.",
        "- The future filter should use only known-at-time liquidation/BBO/trade features and should "
        "evaluate kept turnover against the 500k USD/day constraint.",
        "",
        "## Risks",
        "",
        "- Full EDA stores compact aggregates, not every enriched trade row.",
        "- Liquidation response is descriptive, not causal proof.",
        "- Same-timestamp ordering remains ambiguous in public data.",
        "- A full production signal still needs a separate feature-generation path that returns one filter value per trade.",
        "",
        ]
    )
    if markout_context.height:
        lines.extend(["## Evidence Pointer", "", "See `reports/liquidation_eda/tables/markout_by_liquidation_context.csv`.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_liquidation_eda(args: argparse.Namespace) -> None:
    config = load_liquidation_config(args.config, getattr(args, "profile", None))
    ensure_output_dirs(config)
    metadata: dict[str, Any] = {
        "profile": config.profile,
        "config_path": str(config.config_path) if config.config_path else None,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "polars_version": pl.__version__,
        "numpy_version": _package_version("numpy"),
        "data_scope_note": (
            "Core markout, liquidation context, OFI, queue imbalance, response functions, "
            "and event-study tables are computed on all rows for the selected profile. "
            "Deterministic samples are used only for visual distribution plots."
        ),
        "outputs": [],
    }
    specs = source_specs(config.symbols)

    with _phase(metadata, "source_audit"):
        audit_names = [
            "source_files.csv",
            "schema_audit.csv",
            "time_coverage.csv",
            "side_counts.csv",
            "daily_event_counts.csv",
            "hourly_event_counts.csv",
        ]
        cached_audit = (
            _read_existing_tables(config.tables_dir, audit_names)
            if config.reuse_source_audit
            else None
        )
        if cached_audit is not None:
            metadata["source_audit_reused"] = True
            source_files = cached_audit["source_files.csv"]
            schema = cached_audit["schema_audit.csv"]
            coverage = cached_audit["time_coverage.csv"]
            side_counts = cached_audit["side_counts.csv"]
            daily = cached_audit["daily_event_counts.csv"]
            hourly = cached_audit["hourly_event_counts.csv"]
        else:
            metadata["source_audit_reused"] = False
            source_files = source_file_table(config)
            schema = schema_audit(config)
            coverage, side_counts = source_quality_tables(config)
            daily = daily_event_counts(config)
            hourly = hourly_event_counts(config)
            for name, df in [
                ("source_files.csv", source_files),
                ("schema_audit.csv", schema),
                ("time_coverage.csv", coverage),
                ("side_counts.csv", side_counts),
                ("daily_event_counts.csv", daily),
                ("hourly_event_counts.csv", hourly),
            ]:
                _write_csv(df, config.tables_dir / name)
        metadata["source_rows"] = {
            f"{row['source']}:{row['symbol']}": row["rows"] for row in coverage.to_dicts()
        }

    with _phase(metadata, "full_data_eda"):
        full_outputs = compute_full_data_eda(config, specs)
        metadata.update(full_outputs.metadata)

    bbo_samples: list[pl.DataFrame] = []
    trade_samples: list[pl.DataFrame] = []
    liq_samples: list[pl.DataFrame] = []
    markout_samples: list[pl.DataFrame] = []
    bbo_quality_rows: list[pl.DataFrame] = []
    ofi_rows: list[pl.DataFrame] = []
    qi_rows: list[pl.DataFrame] = []
    trade_summary_rows: list[pl.DataFrame] = []
    liq_summary_rows: list[pl.DataFrame] = []
    price_location_rows: list[pl.DataFrame] = []
    convention_rows: list[pl.DataFrame] = []
    response_rows: list[pl.DataFrame] = []

    with _phase(metadata, "sampled_symbol_eda"):
        for symbol in config.symbols:
            bbo_spec = _spec(specs, BINANCE_BBO, symbol)
            trade_spec = _spec(specs, BINANCE_TRADES, symbol)
            bin_liq_spec = _spec(specs, BINANCE_LIQUIDATIONS, symbol)
            bybit_liq_spec = _spec(specs, BYBIT_LIQUIDATIONS, symbol)

            bbo = _prepare_bbo_sample(
                config, bbo_spec, config.active_profile.max_bbo_rows_per_symbol
            )
            trades = _prepare_trade_sample(
                config, trade_spec, config.active_profile.max_trade_rows_per_symbol
            )
            bin_liq = _prepare_liq_sample(
                config, bin_liq_spec, config.active_profile.max_liquidation_rows_per_symbol
            )
            bybit_liq = _prepare_liq_sample(
                config, bybit_liq_spec, config.active_profile.max_liquidation_rows_per_symbol
            )

            bbo_samples.append(bbo)
            trade_samples.append(trades)
            liq_samples.extend(
                [
                    bin_liq.with_columns(pl.lit("binance").alias("venue")),
                    bybit_liq.with_columns(pl.lit("bybit").alias("venue")),
                ]
            )
            bbo_quality_rows.append(bbo_quality(bbo, symbol))
            ofi_rows.append(bbo_ofi_summary(bbo, symbol))
            qi_rows.append(queue_imbalance_next_move(bbo, symbol))
            trade_summary_rows.append(trade_summary(trades, symbol))
            liq_summary_rows.append(liquidation_summary(bin_liq, symbol, "binance", config.bybit_delay_us))
            liq_summary_rows.append(liquidation_summary(bybit_liq, symbol, "bybit", config.bybit_delay_us))

            rows_per_split = max(1, config.active_profile.max_trade_rows_per_symbol // 2)
            for split in ("train", "validation"):
                trades_window = _sample_split_head(
                    config, trade_spec, split=split, max_rows=rows_per_split
                )
                if trades_window.height == 0:
                    continue
                bbo_dense = _bbo_slice_for_events(config, bbo_spec, trades_window)
                if bbo_dense.height == 0:
                    continue
                joined = join_trades_to_bbo(
                    trades_window, bbo_dense, config.bbo_staleness_tolerance_us
                )
                markouts = compute_markouts(
                    joined,
                    bbo_dense,
                    config.eda_curve_horizons_seconds,
                    config.maker_rebate_bps,
                    config.bbo_staleness_tolerance_us,
                )
                markouts = add_split_column(markouts).with_columns(pl.lit(split).alias("sample_split"))
                bin_liq_window = _liquidation_slice_for_events(
                    config, bin_liq_spec, trades_window, venue="binance"
                )
                bybit_liq_window = _liquidation_slice_for_events(
                    config, bybit_liq_spec, trades_window, venue="bybit"
                )
                combined_liq = pl.concat(
                    [bin_liq_window, bybit_liq_window], how="diagonal_relaxed"
                )
                markouts = _add_recent_liq_pressure(
                    markouts,
                    combined_liq,
                    timestamp_col="available_timestamp",
                    output_col="signed_liq_notional_30s",
                    window_seconds=30,
                )
                markout_samples.append(markouts)
                price_location_rows.append(price_location_summary(markouts, symbol))
                convention_rows.append(convention_examples(markouts, config.task_horizons_seconds))

                trade_events = trades_window.with_columns(
                    [
                        (pl.col("price") * pl.col("amount")).alias("notional"),
                        (
                            pl.when(pl.col("side") == "buy")
                            .then(1)
                            .when(pl.col("side") == "sell")
                            .then(-1)
                            .otherwise(None)
                        ).alias("taker_direction"),
                    ]
                ).with_columns(
                    (pl.col("taker_direction") * pl.col("notional")).alias(
                        "signed_trade_notional"
                    )
                )
                response_rows.append(
                    _response_functions(
                        trade_events,
                        bbo_dense,
                        symbol=symbol,
                        flow_type=f"binance_trade_flow_{split}",
                        timestamp_col="timestamp",
                        signed_flow_col="signed_trade_notional",
                        horizons_seconds=config.eda_curve_horizons_seconds,
                        tolerance_us=config.bbo_staleness_tolerance_us,
                    )
                )
                response_rows.append(
                    _response_functions(
                        bin_liq_window,
                        bbo_dense,
                        symbol=symbol,
                        flow_type=f"binance_liquidation_flow_{split}",
                        timestamp_col="available_timestamp",
                        signed_flow_col="signed_liquidation_notional",
                        horizons_seconds=config.eda_curve_horizons_seconds,
                        tolerance_us=config.bbo_staleness_tolerance_us,
                    )
                )
                response_rows.append(
                    _response_functions(
                        bybit_liq_window,
                        bbo_dense,
                        symbol=symbol,
                        flow_type=f"bybit_liquidation_flow_available_{split}",
                        timestamp_col="available_timestamp",
                        signed_flow_col="signed_liquidation_notional",
                        horizons_seconds=config.eda_curve_horizons_seconds,
                        tolerance_us=config.bbo_staleness_tolerance_us,
                    )
                )

    with _phase(metadata, "write_tables"):
        full_tables = full_outputs.tables
        all_bbo = _concat_or_empty(bbo_samples)
        all_trades = _concat_or_empty(trade_samples)
        all_liq = _concat_or_empty(liq_samples)
        all_markouts = _concat_or_empty(markout_samples)
        bbo_quality_df = full_tables.get("bbo_quality", _concat_or_empty(bbo_quality_rows))
        price_location_df = full_tables.get(
            "trade_price_location_summary", _concat_or_empty(price_location_rows)
        )
        markout_summary = full_tables.get(
            "full_markout_summary",
            summarize_markouts(all_markouts, config.eda_curve_horizons_seconds),
        )
        task_markout_summary = (
            markout_summary.filter(pl.col("horizon_seconds").is_in(list(config.task_horizons_seconds)))
            if "horizon_seconds" in markout_summary.columns
            else pl.DataFrame()
        )
        daily_markout = full_tables.get(
            "full_daily_weighted_markout",
            daily_stability(all_markouts, config.task_horizons_seconds),
        )
        markout_context = full_tables.get(
            "full_markout_by_liquidation_context",
            _markout_by_liquidation_context(all_markouts, config.task_horizons_seconds),
        )
        nonlinear = full_tables.get(
            "nonlinear_flow_response",
            nonlinear_flow_response(all_markouts, config.task_horizons_seconds),
        )
        responses = full_tables.get(
            "signed_flow_response_functions",
            _concat_or_empty([df for df in response_rows if df.height > 0]),
        )
        event_study = full_tables.get("event_study_summary", pl.DataFrame())
        asof_sensitivity = full_tables.get("asof_sensitivity", pl.DataFrame())
        train_validation = (
            daily.filter(
                pl.col("date").is_between(pl.lit("2025-12-01"), pl.lit("2026-02-28"))
            )
            .with_columns(
                (
                    pl.when(pl.col("date") <= pl.lit("2026-01-31"))
                    .then(pl.lit("train"))
                    .otherwise(pl.lit("validation"))
                ).alias("split")
            )
            .group_by(["source", "symbol", "split"])
            .agg(pl.col("rows").sum().alias("rows"))
        )
        anomaly = _anomaly_log(bbo_quality_df, price_location_df, coverage)
        table_map = {
            "bbo_quality.csv": bbo_quality_df,
            "bbo_ofi_summary.csv": full_tables.get("bbo_ofi_summary", _concat_or_empty(ofi_rows)),
            "queue_imbalance_next_move.csv": full_tables.get(
                "queue_imbalance_next_move", _concat_or_empty(qi_rows)
            ),
            "trade_summary.csv": full_tables.get("trade_summary", _concat_or_empty(trade_summary_rows)),
            "liquidation_summary.csv": full_tables.get(
                "liquidation_summary", _concat_or_empty(liq_summary_rows)
            ),
            "trade_price_location_summary.csv": price_location_df,
            "trade_side_bbo_diagnostic.csv": price_location_df,
            "convention_examples_trades.csv": _concat_or_empty(convention_rows),
            "markout_summary.csv": markout_summary,
            "baseline_all_trades_markout.csv": task_markout_summary,
            "daily_weighted_markout.csv": daily_markout,
            "markout_by_liquidation_context.csv": markout_context,
            "full_markout_summary.csv": markout_summary,
            "full_daily_weighted_markout.csv": daily_markout,
            "full_markout_by_liquidation_context.csv": markout_context,
            "train_validation_drift.csv": train_validation,
            "signed_flow_response_functions.csv": responses,
            "nonlinear_flow_response.csv": nonlinear,
            "event_study_summary.csv": event_study,
            "asof_sensitivity.csv": asof_sensitivity,
            "anomaly_log.csv": anomaly,
        }
        for name, df in table_map.items():
            _write_csv(df, config.tables_dir / name)
        all_bbo.write_parquet(config.processed_root / f"bbo_sample_{config.profile}.parquet")
        all_trades.write_parquet(config.processed_root / f"trade_sample_{config.profile}.parquet")
        all_liq.write_parquet(config.processed_root / f"liquidation_sample_{config.profile}.parquet")
        all_markouts.write_parquet(config.processed_root / f"markout_sample_{config.profile}.parquet")

    with _phase(metadata, "figures"):
        plot_event_counts_by_day(daily, config.figures_dir / "event_counts_by_day.png")
        plot_hist(
            all_bbo,
            "spread_bps",
            config.figures_dir / "spread_distribution_bps.png",
            "Sampled BBO Spread, p99.5 clipped",
            xlabel="spread, bps",
            log_y=True,
            clip_quantile=0.995,
        )
        trade_plot = all_trades.with_columns(
            (pl.col("price") * pl.col("amount")).alias("notional")
        ).with_columns(
            pl.col("notional").log10().alias("log10_notional")
        )
        plot_hist(
            trade_plot,
            "log10_notional",
            config.figures_dir / "trade_notional_distribution.png",
            "Sampled Trade Notional, log10 USD",
            xlabel="log10(notional USD)",
        )
        liq_plot = all_liq.with_columns(pl.col("notional").log10().alias("log10_notional"))
        plot_hist(
            liq_plot,
            "log10_notional",
            config.figures_dir / "liquidation_notional_distribution.png",
            "Sampled Liquidation Notional, log10 USD",
            xlabel="log10(notional USD)",
        )
        plot_ofi_response(
            table_map["bbo_ofi_summary.csv"], config.figures_dir / "ofi_vs_future_return.png"
        )
        plot_queue_imbalance(
            table_map["queue_imbalance_next_move.csv"],
            config.figures_dir / "queue_imbalance_next_move_probability.png",
        )
        plot_markout_distribution(all_markouts, config.task_horizons_seconds[0], config.figures_dir / "markout_distribution_by_tau.png")
        plot_markout_curve(markout_summary, config.figures_dir / "markout_curve_by_side_symbol.png")
        plot_response_functions(responses, config.figures_dir / "signed_flow_response_functions.png")
        plot_nonlinear_response(nonlinear, config.figures_dir / "nonlinear_flow_response.png")
        plot_event_study(
            event_study, config.figures_dir / "liquidation_event_study_mid.png", detailed=False
        )
        plot_event_study(
            event_study,
            config.figures_dir / "liquidation_event_study_by_venue_side_symbol.png",
            detailed=True,
        )

    with _phase(metadata, "report"):
        _write_hypotheses(config, markout_context)
        metadata["outputs"] = [
            str(config.output_root / "liquidation_eda_report.md"),
            str(config.output_root / "run_metadata.json"),
        ]
        metadata_path = config.output_root / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        write_liquidation_report(config.output_root / "liquidation_eda_report.md", metadata)
    print(f"Wrote {config.output_root / 'liquidation_eda_report.md'}")
