from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cmf_backtester.data.schema import required_l1_columns, validate_columns


@dataclass(frozen=True)
class MarketDataArrays:
    timestamps: np.ndarray
    best_bid_ticks: np.ndarray
    best_ask_ticks: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    mid_ticks: np.ndarray
    mid_half_ticks: np.ndarray
    spread_ticks: np.ndarray
    imbalance: np.ndarray
    split: np.ndarray
    date: np.ndarray
    tick_size: float

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def subset(self, mask: np.ndarray) -> "MarketDataArrays":
        return MarketDataArrays(
            timestamps=self.timestamps[mask],
            best_bid_ticks=self.best_bid_ticks[mask],
            best_ask_ticks=self.best_ask_ticks[mask],
            bid_size=self.bid_size[mask],
            ask_size=self.ask_size[mask],
            mid_ticks=self.mid_ticks[mask],
            mid_half_ticks=self.mid_half_ticks[mask],
            spread_ticks=self.spread_ticks[mask],
            imbalance=self.imbalance[mask],
            split=self.split[mask],
            date=self.date[mask],
            tick_size=self.tick_size,
        )


def scan_lob_csv(path: str | Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema_length=1000)
    validate_columns(lf.collect_schema().names(), required_l1_columns())
    return lf


def scan_trades_csv(path: str | Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema_length=1000)
    validate_columns(lf.collect_schema().names(), ["local_timestamp", "side", "price", "amount"])
    return lf


def load_lob_l1(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_processed_arrays(path: str | Path, tick_size: float) -> MarketDataArrays:
    df = load_lob_l1(path)
    required = [
        "event_id",
        "local_timestamp",
        "best_bid_ticks",
        "best_ask_ticks",
        "bid_size",
        "ask_size",
        "mid_ticks",
        "mid_half_ticks",
        "spread_ticks",
        "imbalance",
        "split",
        "date",
    ]
    validate_columns(df.columns, required)
    return MarketDataArrays(
        timestamps=df["local_timestamp"].to_numpy().astype(np.int64),
        best_bid_ticks=df["best_bid_ticks"].to_numpy().astype(np.int64),
        best_ask_ticks=df["best_ask_ticks"].to_numpy().astype(np.int64),
        bid_size=df["bid_size"].to_numpy().astype(np.float64),
        ask_size=df["ask_size"].to_numpy().astype(np.float64),
        mid_ticks=df["mid_ticks"].to_numpy().astype(np.float64),
        mid_half_ticks=df["mid_half_ticks"].to_numpy().astype(np.int64),
        spread_ticks=df["spread_ticks"].to_numpy().astype(np.int64),
        imbalance=df["imbalance"].to_numpy().astype(np.float64),
        split=df["split"].to_numpy().astype(str),
        date=df["date"].to_numpy().astype(str),
        tick_size=float(tick_size),
    )
