from __future__ import annotations

import pandas as pd

from execsim.data.cleaning import clean_intraday_bars


def test_clean_intraday_bars_filters_regular_hours_and_deduplicates() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "timestamp": [
                "2026-03-16T13:29:00Z",
                "2026-03-16T13:30:00Z",
                "2026-03-16T13:30:00Z",
                "2026-03-16T14:00:00Z",
                "2026-03-16T20:00:00Z",
            ],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "close": [100.2, 101.2, 102.2, 103.2, 104.2],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "trade_count": [10, 11, 12, 13, 14],
            "vwap": [100.1, 101.1, 102.1, 103.1, 104.1],
            "extra": [1, 2, 3, 4, 5],
        }
    )

    cleaned = clean_intraday_bars(raw, timezone="America/New_York")

    assert list(cleaned.columns) == [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    assert len(cleaned) == 2
    assert cleaned["timestamp"].dt.strftime("%H:%M").tolist() == ["09:30", "10:00"]
    assert cleaned["close"].tolist() == [102.2, 103.2]
    assert cleaned["symbol"].tolist() == ["AAPL", "AAPL"]


def test_clean_intraday_bars_accepts_symbol_timestamp_index() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "timestamp": [pd.Timestamp("2026-03-16T13:30:00Z")],
            "open": [200.0],
            "high": [201.0],
            "low": [199.0],
            "close": [200.5],
            "volume": [500],
            "trade_count": [5],
            "vwap": [200.4],
        }
    ).set_index(["symbol", "timestamp"])

    cleaned = clean_intraday_bars(raw, timezone="America/New_York")

    assert len(cleaned) == 1
    assert cleaned.loc[0, "symbol"] == "MSFT"
    assert cleaned.loc[0, "timestamp"].strftime("%H:%M") == "09:30"
