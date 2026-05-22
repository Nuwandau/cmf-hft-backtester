from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmf_backtester.utils.config import load_config


@dataclass(frozen=True)
class ProfileConfig:
    max_trade_rows_per_symbol: int
    max_bbo_rows_per_symbol: int
    max_liquidation_rows_per_symbol: int
    event_study_top_n: int
    full_data_batch_minutes: int
    max_full_data_batches_per_date: int


@dataclass(frozen=True)
class LiquidationEdaConfig:
    raw_root: Path
    output_root: Path
    processed_root: Path
    symbols: tuple[str, ...]
    task_horizons_seconds: tuple[int, ...]
    eda_curve_horizons_seconds: tuple[int, ...]
    short_flow_windows_ms: tuple[int, ...]
    bybit_delay_us: int
    maker_rebate_bps: float
    bbo_staleness_tolerance_us: int
    plot_sample_rows: int
    reuse_source_audit: bool
    profile: str
    quick: ProfileConfig
    full: ProfileConfig
    config_path: Path | None = None

    @property
    def active_profile(self) -> ProfileConfig:
        return self.quick if self.profile == "quick" else self.full

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    def with_profile(self, profile: str | None) -> "LiquidationEdaConfig":
        if profile is None or profile == self.profile:
            return self
        if profile not in {"quick", "full"}:
            raise ValueError(f"Unsupported liquidation EDA profile: {profile}")
        return LiquidationEdaConfig(
            raw_root=self.raw_root,
            output_root=self.output_root,
            processed_root=self.processed_root,
            symbols=self.symbols,
            task_horizons_seconds=self.task_horizons_seconds,
            eda_curve_horizons_seconds=self.eda_curve_horizons_seconds,
            short_flow_windows_ms=self.short_flow_windows_ms,
            bybit_delay_us=self.bybit_delay_us,
            maker_rebate_bps=self.maker_rebate_bps,
            bbo_staleness_tolerance_us=self.bbo_staleness_tolerance_us,
            plot_sample_rows=self.plot_sample_rows,
            reuse_source_audit=self.reuse_source_audit,
            profile=profile,
            quick=self.quick,
            full=self.full,
            config_path=self.config_path,
        )


def _tuple_int(raw: Any, default: list[int]) -> tuple[int, ...]:
    values = default if raw is None else raw
    return tuple(int(v) for v in values)


def _tuple_str(raw: Any, default: list[str]) -> tuple[str, ...]:
    values = default if raw is None else raw
    return tuple(str(v).lower() for v in values)


def _profile(raw: dict[str, Any] | None, defaults: dict[str, int]) -> ProfileConfig:
    raw = raw or {}
    return ProfileConfig(
        max_trade_rows_per_symbol=int(
            raw.get("max_trade_rows_per_symbol", defaults["max_trade_rows_per_symbol"])
        ),
        max_bbo_rows_per_symbol=int(
            raw.get("max_bbo_rows_per_symbol", defaults["max_bbo_rows_per_symbol"])
        ),
        max_liquidation_rows_per_symbol=int(
            raw.get(
                "max_liquidation_rows_per_symbol",
                defaults["max_liquidation_rows_per_symbol"],
            )
        ),
        event_study_top_n=int(raw.get("event_study_top_n", defaults["event_study_top_n"])),
        full_data_batch_minutes=int(
            raw.get("full_data_batch_minutes", defaults["full_data_batch_minutes"])
        ),
        max_full_data_batches_per_date=int(
            raw.get(
                "max_full_data_batches_per_date",
                defaults["max_full_data_batches_per_date"],
            )
        ),
    )


def load_liquidation_config(path: str | Path, profile: str | None = None) -> LiquidationEdaConfig:
    raw = load_config(path)
    default_quick = {
        "max_trade_rows_per_symbol": 50_000,
        "max_bbo_rows_per_symbol": 75_000,
        "max_liquidation_rows_per_symbol": 25_000,
        "event_study_top_n": 250,
        "full_data_batch_minutes": 15,
        "max_full_data_batches_per_date": 2,
    }
    default_full = {
        "max_trade_rows_per_symbol": 300_000,
        "max_bbo_rows_per_symbol": 300_000,
        "max_liquidation_rows_per_symbol": 100_000,
        "event_study_top_n": 2_000,
        "full_data_batch_minutes": 60,
        "max_full_data_batches_per_date": 0,
    }
    cfg = LiquidationEdaConfig(
        raw_root=Path(raw.get("raw_root", "data/raw/liquidation_task")),
        output_root=Path(raw.get("output_root", "reports/liquidation_eda")),
        processed_root=Path(raw.get("processed_root", "data/processed/liquidation_task")),
        symbols=_tuple_str(raw.get("symbols"), ["btcusdt", "ethusdt"]),
        task_horizons_seconds=_tuple_int(raw.get("markout_horizons_seconds"), [30, 120, 300]),
        eda_curve_horizons_seconds=_tuple_int(
            raw.get("eda_curve_horizons_seconds"), [1, 5, 10, 30, 60, 120, 300]
        ),
        short_flow_windows_ms=_tuple_int(
            raw.get("short_flow_windows_ms"), [100, 250, 500, 1000, 2000]
        ),
        bybit_delay_us=int(raw.get("bybit_delay_us", 200_000)),
        maker_rebate_bps=float(raw.get("maker_rebate_bps", 0.5)),
        bbo_staleness_tolerance_us=int(raw.get("bbo_staleness_tolerance_us", 5_000_000)),
        plot_sample_rows=int(raw.get("plot_sample_rows", 200_000)),
        reuse_source_audit=bool(raw.get("reuse_source_audit", True)),
        profile=str(raw.get("profile", "quick")),
        quick=_profile(raw.get("quick"), default_quick),
        full=_profile(raw.get("full"), default_full),
        config_path=Path(path),
    )
    return cfg.with_profile(profile)
