from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


BINANCE_TRADES = "binance_trades"
BINANCE_BBO = "binance_booktickers"
BINANCE_LIQUIDATIONS = "binance_liquidations"
BYBIT_LIQUIDATIONS = "bybit_liquidations"


@dataclass(frozen=True)
class SourceSpec:
    source: str
    venue: str
    data_type: str
    symbol: str
    ticker: str
    relative_path: Path
    columns: tuple[str, ...]


TRADE_COLUMNS = ("timestamp", "ticker", "side", "price", "amount")
BBO_COLUMNS = ("timestamp", "ticker", "bid_price", "bid_amount", "ask_price", "ask_amount")


EXPECTED_DTYPES = {
    "timestamp": pl.Int64,
    "ticker": pl.String,
    "side": pl.String,
    "price": pl.Float64,
    "amount": pl.Float64,
    "bid_price": pl.Float64,
    "bid_amount": pl.Float64,
    "ask_price": pl.Float64,
    "ask_amount": pl.Float64,
}


def binance_ticker(symbol: str) -> str:
    return f"perp:{symbol}"


def source_specs(symbols: tuple[str, ...]) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for symbol in symbols:
        specs.extend(
            [
                SourceSpec(
                    source=BINANCE_TRADES,
                    venue="binance",
                    data_type="trades",
                    symbol=symbol,
                    ticker=binance_ticker(symbol),
                    relative_path=Path("data/binance_trades") / f"perp_{symbol}.parquet",
                    columns=TRADE_COLUMNS,
                ),
                SourceSpec(
                    source=BINANCE_BBO,
                    venue="binance",
                    data_type="bbo",
                    symbol=symbol,
                    ticker=binance_ticker(symbol),
                    relative_path=Path("data/binance_booktickers") / f"perp_{symbol}.parquet",
                    columns=BBO_COLUMNS,
                ),
                SourceSpec(
                    source=BINANCE_LIQUIDATIONS,
                    venue="binance",
                    data_type="liquidations",
                    symbol=symbol,
                    ticker=binance_ticker(symbol),
                    relative_path=Path("data/binance_liquidations") / f"perp_{symbol}.parquet",
                    columns=TRADE_COLUMNS,
                ),
                SourceSpec(
                    source=BYBIT_LIQUIDATIONS,
                    venue="bybit",
                    data_type="liquidations",
                    symbol=symbol,
                    ticker=symbol,
                    relative_path=Path("data/bybit_liquidations") / f"{symbol}.parquet",
                    columns=TRADE_COLUMNS,
                ),
            ]
        )
    return specs


def maker_direction_expr() -> pl.Expr:
    return (
        pl.when(pl.col("side") == "buy")
        .then(1)
        .when(pl.col("side") == "sell")
        .then(-1)
        .otherwise(None)
        .alias("maker_direction")
    )


def liquidation_direction_expr() -> pl.Expr:
    return (
        pl.when(pl.col("side") == "buy")
        .then(1)
        .when(pl.col("side") == "sell")
        .then(-1)
        .otherwise(None)
        .alias("liquidation_direction")
    )
