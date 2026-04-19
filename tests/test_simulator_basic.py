from __future__ import annotations

from datetime import date, time
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execsim.orders import ParentOrder
from execsim.simulator import simulate_twap


def test_simulator_respects_participation_cap_and_reports_incomplete_fill() -> None:
    bars = _make_bars(
        n_bars=3,
        volume=[100, 100, 100],
        vwap=[10.0, 20.0, 30.0],
    )
    order = _make_order(side="buy", quantity=60, end_time=time(9, 33))

    result = simulate_twap(order, bars, max_bar_participation_rate=0.10)

    assert result.execution_log["scheduled_qty"].tolist() == [20, 20, 20]
    assert result.execution_log["max_allowed_qty"].tolist() == [10, 10, 10]
    assert result.execution_log["filled_qty"].tolist() == [10, 10, 10]
    assert result.summary.filled_qty == 30
    assert result.summary.unfilled_qty == 30
    assert result.summary.completion_rate == pytest.approx(0.5)
    assert result.summary.realized_participation == pytest.approx(0.1)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_simulator_handles_buy_and_sell(side: str) -> None:
    bars = _make_bars(
        n_bars=2,
        volume=[1_000, 1_000],
        vwap=[10.0, 11.0],
    )
    order = _make_order(side=side, quantity=4, end_time=time(9, 32))

    result = simulate_twap(order, bars, max_bar_participation_rate=1.0)

    assert result.summary.side == side
    assert result.summary.filled_qty == 4
    assert result.summary.unfilled_qty == 0
    assert result.execution_log["side"].tolist() == [side, side]


def test_simulator_average_fill_price_uses_vwap_then_bar_proxy_fallback() -> None:
    bars = _make_bars(
        n_bars=2,
        volume=[1_000, 1_000],
        open_=[10.0, 20.0],
        high=[10.0, 22.0],
        low=[10.0, 18.0],
        close=[10.0, 20.0],
        vwap=[10.0, None],
    )
    order = _make_order(side="buy", quantity=4, end_time=time(9, 32))

    result = simulate_twap(order, bars, max_bar_participation_rate=1.0)

    assert result.execution_log["fill_price"].tolist() == [10.0, 20.0]
    assert result.summary.average_fill_price == pytest.approx(15.0)


def _make_order(side: str, quantity: int, end_time: time) -> ParentOrder:
    return ParentOrder(
        symbol="AAPL",
        side=side,
        quantity=quantity,
        trade_date=date(2026, 3, 16),
        start_time=time(9, 30),
        end_time=end_time,
    )


def _make_bars(
    n_bars: int,
    volume: list[int],
    vwap: list[float | None],
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
) -> pd.DataFrame:
    open_ = open_ or [10.0 + index for index in range(n_bars)]
    high = high or [value + 0.5 for value in open_]
    low = low or [value - 0.5 for value in open_]
    close = close or open_

    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * n_bars,
            "timestamp": pd.date_range(
                "2026-03-16 09:30",
                periods=n_bars,
                freq="min",
                tz="America/New_York",
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": [10] * n_bars,
            "vwap": vwap,
        }
    )
