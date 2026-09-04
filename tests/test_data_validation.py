from __future__ import annotations

import pandas as pd

from execsim.data.validation import validate_processed_bars


def test_validation_flags_duplicate_and_non_positive_volume() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                [
                    "2026-03-16T09:30:00-04:00",
                    "2026-03-16T09:30:00-04:00",
                    "2026-03-16T09:31:00-04:00",
                ]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.2, 101.2, 102.2],
            "volume": [100, 200, 0],
            "trade_count": [10, 11, 12],
            "vwap": [100.1, 101.1, 102.1],
        }
    )

    report = validate_processed_bars(bars, symbol="AAPL")

    assert report.is_valid is False
    assert report.duplicate_timestamps == 1
    assert report.non_increasing_timestamps == 1
    assert report.non_positive_volume_rows == 1
    assert report.non_full_days[0].bar_count == 3


def test_validation_flags_missing_required_columns() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": pd.to_datetime(["2026-03-16T09:30:00-04:00"]),
            "open": [100.0],
            "high": [100.5],
            "low": [99.5],
            "close": [100.2],
            "volume": [100],
            "trade_count": [10],
        }
    )

    report = validate_processed_bars(bars, symbol="AAPL")

    assert report.is_valid is False
    assert report.missing_columns == ("vwap",)
