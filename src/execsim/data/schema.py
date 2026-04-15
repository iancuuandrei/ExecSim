from __future__ import annotations

from datetime import time

BAR_COLUMNS = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)
REQUIRED_COLUMNS = BAR_COLUMNS
REQUIRED_NON_NULL_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
FULL_TRADING_DAY_BAR_COUNT = 390
