from __future__ import annotations

import pandas as pd

from execsim.data.schema import BAR_COLUMNS, MARKET_CLOSE_TIME, MARKET_OPEN_TIME


def clean_intraday_bars(raw_bars: pd.DataFrame, timezone: str) -> pd.DataFrame:
    bars = _ensure_symbol_timestamp_columns(raw_bars)
    missing_columns = [column for column in BAR_COLUMNS if column not in bars.columns]
    if missing_columns:
        raise ValueError(f"Raw bars missing required columns: {missing_columns}")

    cleaned = bars.loc[:, BAR_COLUMNS].copy()
    cleaned["symbol"] = cleaned["symbol"].astype(str).str.upper()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True).dt.tz_convert(timezone)
    cleaned = cleaned.sort_values(["symbol", "timestamp"], kind="stable")
    cleaned = cleaned.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    cleaned = _filter_regular_hours(cleaned)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def _ensure_symbol_timestamp_columns(raw_bars: pd.DataFrame) -> pd.DataFrame:
    bars = raw_bars.copy()
    if "symbol" in bars.columns and "timestamp" in bars.columns:
        return bars

    bars = bars.reset_index()
    if "symbol" not in bars.columns or "timestamp" not in bars.columns:
        raise ValueError("Raw bars must include symbol and timestamp columns or index levels.")

    return bars


def _filter_regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    timestamps = bars["timestamp"].dt.time
    mask = (timestamps >= MARKET_OPEN_TIME) & (timestamps < MARKET_CLOSE_TIME)
    return bars.loc[mask].copy()
