from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cmf_backtester.reporting.report import _format_value, _polars_to_markdown


def _read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path) if path.exists() else pl.DataFrame()


def _add_table(lines: list[str], title: str, path: Path, cols: list[str], head: int = 12) -> None:
    if not path.exists():
        return
    df = pl.read_csv(path)
    available = [col for col in cols if col in df.columns]
    if not available:
        return
    lines.extend([f"## {title}", "", _polars_to_markdown(df.select(available).head(head)), ""])
    lines.append(f"Full table: `{path.as_posix()}`")
    lines.append("")


def _add_figure(lines: list[str], title: str, path: Path) -> None:
    if path.exists():
        try:
            figure_ref = path.relative_to(path.parent.parent).as_posix()
        except ValueError:
            figure_ref = path.as_posix()
        lines.extend([f"### {title}", "", f"![{title}]({figure_ref})", ""])


def _key_findings(tables_dir: Path) -> list[str]:
    findings: list[str] = []
    coverage = _read(tables_dir / "time_coverage.csv")
    if coverage.height:
        rows = int(coverage["rows"].sum())
        findings.append(f"Total raw rows across sources: `{rows:,}`.")
    markout = _read(tables_dir / "markout_summary.csv")
    if markout.height:
        best = markout.sort("weighted_pnl_bps", descending=True).row(0, named=True)
        worst = markout.sort("weighted_pnl_bps").row(0, named=True)
        findings.append(
            "Best full-data maker markout bucket: "
            f"`{best.get('symbol')} {best.get('side')} {best.get('horizon_seconds')}s`, "
            f"`{_format_value(best.get('weighted_pnl_bps'))}` bps."
        )
        findings.append(
            "Worst full-data maker markout bucket: "
            f"`{worst.get('symbol')} {worst.get('side')} {worst.get('horizon_seconds')}s`, "
            f"`{_format_value(worst.get('weighted_pnl_bps'))}` bps."
        )
    price_loc = _read(tables_dir / "trade_price_location_summary.csv")
    if price_loc.height:
        findings.append(
            "Trade-side convention is diagnosed by comparing trade prices to previous BBO; "
            "see `trade_price_location_summary.csv`."
        )
    return findings


def write_liquidation_report(output_path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(output_path)
    tables_dir = output_path.parent / "tables"
    figures_dir = output_path.parent / "figures"
    lines: list[str] = [
        "# Liquidation Dataset EDA Report",
        "",
        "## Objective",
        "",
        "Explore Binance trades, Binance BBO, Binance liquidations, and Bybit liquidations "
        "before building any trade-filter signal. The report focuses on conventions, data "
        "quality, markouts, liquidation context, and cross-source relationships.",
        "",
        "## Execution Profile",
        "",
        f"- Profile: `{metadata.get('profile')}`",
        f"- Generated at UTC: `{metadata.get('generated_at_utc')}`",
        f"- Config: `{metadata.get('config_path')}`",
        f"- Data scope: `{metadata.get('data_scope_note')}`",
        f"- Full processed trade rows: `{metadata.get('processed_trade_rows_full')}`",
        f"- Full processed BBO rows: `{metadata.get('processed_bbo_rows_full')}`",
        "",
    ]
    findings = _key_findings(tables_dir)
    if findings:
        lines.extend(["## Key Findings", ""])
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")

    lines.extend(
        [
            "## Convention Checks",
            "",
            "- Timestamps are interpreted as microseconds since UNIX epoch UTC.",
            "- Binance trade `side` is treated as taker side.",
            "- Liquidation `side` is treated as liquidation order side.",
            "- Bybit liquidation features use `available_timestamp = timestamp + 200_000`.",
            "- Known-at-time joins are backward as-of joins; future markouts join to BBO at "
            "`trade_timestamp + tau` using backward fill.",
            "",
        ]
    )

    _add_table(lines, "Source Files", tables_dir / "source_files.csv", ["source", "symbol", "size_mb", "exists"])
    _add_table(
        lines,
        "Time Coverage",
        tables_dir / "time_coverage.csv",
        ["source", "symbol", "rows", "min_datetime_utc", "max_datetime_utc", "duplicate_timestamp_rows"],
    )
    _add_table(
        lines,
        "BBO Quality",
        tables_dir / "bbo_quality.csv",
        ["split", "date", "symbol", "rows", "crossed_rows", "locked_rows", "mean_spread_bps", "mean_queue_imbalance"],
    )
    _add_table(
        lines,
        "Trade Side vs BBO Location",
        tables_dir / "trade_price_location_summary.csv",
        ["split", "symbol", "side", "price_location", "rows", "share_within_side"],
    )
    _add_table(
        lines,
        "Liquidation Summary",
        tables_dir / "liquidation_summary.csv",
        ["split", "date", "venue", "symbol", "side", "rows", "mean_notional", "notional_sum", "clipped_turnover"],
    )
    _add_table(
        lines,
        "Baseline Maker Markout",
        tables_dir / "baseline_all_trades_markout.csv",
        [
            "split",
            "symbol",
            "side",
            "horizon_seconds",
            "rows",
            "weighted_pnl_bps",
            "median_pnl_bps",
            "clipped_turnover",
        ],
    )
    _add_table(
        lines,
        "Markout By Liquidation Context",
        tables_dir / "markout_by_liquidation_context.csv",
        [
            "symbol",
            "side",
            "horizon_seconds",
            "liq_pressure_bucket",
            "venue",
            "window_seconds",
            "maker_vs_liq_pressure",
            "rows",
            "weighted_pnl_bps",
        ],
    )
    _add_table(
        lines,
        "Train Validation Drift",
        tables_dir / "train_validation_drift.csv",
        ["source", "symbol", "split", "rows"],
    )
    _add_table(
        lines,
        "Signed Flow Response",
        tables_dir / "signed_flow_response_functions.csv",
        ["split", "symbol", "flow_type", "horizon_seconds", "response_bps", "rows", "abs_signed_flow_musd"],
    )
    _add_table(
        lines,
        "Nonlinear Liquidation Pressure Buckets",
        tables_dir / "nonlinear_flow_response.csv",
        [
            "symbol",
            "side",
            "venue",
            "window_seconds",
            "horizon_seconds",
            "signed_liq_bucket",
            "rows",
            "weighted_pnl_bps",
        ],
    )
    _add_table(
        lines,
        "As-Of Sensitivity",
        tables_dir / "asof_sensitivity.csv",
        ["asof_mode", "split", "date", "symbol", "side", "price_location", "rows", "share_within_side"],
        head=16,
    )
    _add_table(
        lines,
        "Liquidation Event Study",
        tables_dir / "event_study_summary.csv",
        ["split", "symbol", "venue", "side", "offset_seconds", "rows", "mean_return_bps"],
        head=16,
    )
    _add_table(
        lines,
        "Anomaly Log",
        tables_dir / "anomaly_log.csv",
        ["symbol", "source", "issue_type", "severity", "notes"],
        head=20,
    )

    lines.extend(["## Figures", ""])
    for title, name in [
        ("Events By Day", "event_counts_by_day.png"),
        ("Spread Distribution", "spread_distribution_bps.png"),
        ("Trade Notional Distribution", "trade_notional_distribution.png"),
        ("Liquidation Notional Distribution", "liquidation_notional_distribution.png"),
        ("OFI vs Future Return", "ofi_vs_future_return.png"),
        ("Queue Imbalance Next Move", "queue_imbalance_next_move_probability.png"),
        ("Markout Distribution", "markout_distribution_by_tau.png"),
        ("Markout Curve", "markout_curve_by_side_symbol.png"),
        ("Signed Flow Response", "signed_flow_response_functions.png"),
        ("Nonlinear Flow Response", "nonlinear_flow_response.png"),
        ("Liquidation Event Study", "liquidation_event_study_mid.png"),
        ("Liquidation Event Study By Venue Side Symbol", "liquidation_event_study_by_venue_side_symbol.png"),
    ]:
        _add_figure(lines, title, figures_dir / name)

    lines.extend(
        [
            "## Research Hypotheses For Future Signal",
            "",
            "- Filter trades where maker direction is aligned against recent same-direction "
            "liquidation pressure.",
            "- Bybit liquidation clusters may be more informative than isolated prints after "
            "the required 200ms availability delay.",
            "- OFI and queue imbalance may help separate toxic from normal trades in short "
            "horizons.",
            "- Liquidation impact may be nonlinear: extreme clusters can saturate or reverse.",
            "- Cross-asset stress between BTC and ETH should be tested before deciding whether "
            "signals are symbol-specific.",
            "",
            "## Limitations",
            "",
            "- Core markout and liquidation-context tables are full-data aggregates, not per-trade "
            "materializations. Deterministic samples are still used for visual distribution plots.",
            "- This is not yet a final filter and does not optimize hidden-test score.",
            "- Same-timestamp event ordering cannot be fully recovered from public parquet files.",
            "",
            "## Run Metadata",
            "",
            "```json",
            json.dumps(metadata, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
