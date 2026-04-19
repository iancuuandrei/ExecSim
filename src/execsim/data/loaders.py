from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Iterator

import pandas as pd

from execsim.config import ExecSimConfig


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def load_raw_symbol_bars(config: ExecSimConfig, symbol: str) -> pd.DataFrame:
    path = config.raw_symbol_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Raw parquet not found for {symbol}: {path}")
    return load_parquet(path)


def load_processed_symbol_bars(config: ExecSimConfig, symbol: str) -> pd.DataFrame:
    path = config.processed_symbol_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found for {symbol}: {path}")
    return load_parquet(path)


def load_processed_symbol_day_bars(
    config: ExecSimConfig,
    symbol: str,
    trade_date: date,
) -> pd.DataFrame:
    bars = load_processed_symbol_bars(config, symbol)
    return slice_processed_symbol_bars(
        bars=bars,
        symbol=symbol,
        trade_date=trade_date,
        start_time=None,
        end_time=None,
    )


def load_processed_window_bars(
    config: ExecSimConfig,
    symbol: str,
    trade_date: date,
    start_time: time,
    end_time: time,
) -> pd.DataFrame:
    bars = load_processed_symbol_bars(config, symbol)
    return slice_processed_symbol_bars(
        bars=bars,
        symbol=symbol,
        trade_date=trade_date,
        start_time=start_time,
        end_time=end_time,
    )


def slice_processed_symbol_bars(
    bars: pd.DataFrame,
    symbol: str,
    trade_date: date,
    start_time: time | None = None,
    end_time: time | None = None,
) -> pd.DataFrame:
    if "timestamp" not in bars.columns:
        raise ValueError("Processed bars must include a timestamp column.")

    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start_time must be before end_time.")

    sliced = bars.copy()
    sliced["timestamp"] = pd.to_datetime(sliced["timestamp"])

    if "symbol" in sliced.columns:
        sliced = sliced.loc[
            sliced["symbol"].astype(str).str.upper() == symbol.upper()
        ].copy()

    timestamps = sliced["timestamp"]
    mask = timestamps.dt.date == trade_date

    if start_time is not None:
        mask &= timestamps.dt.time >= start_time
    if end_time is not None:
        mask &= timestamps.dt.time < end_time

    return (
        sliced.loc[mask]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )


def iter_processed_symbol_paths(config: ExecSimConfig) -> Iterator[tuple[str, Path]]:
    for symbol in config.symbols:
        yield symbol, config.processed_symbol_path(symbol)
