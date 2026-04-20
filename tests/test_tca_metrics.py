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


@pytest.mark.parametrize(
    ("side", "expected_shortfall_bps"),
    [
        ("buy", 1000.0),
        ("sell", -1000.0),
    ],
)
def test_implementation_shortfall_sign_follows_order_side(
    side: str,
    expected_shortfall_bps: float,
) -> None:
    bars = _make_bars(
        volume=[1_000, 1_000],
        vwap=[10.0, 12.0],
    )
    order = _make_order(side=side, quantity=2, start_time=time(9, 30), end_time=time(9, 32))

    result = simulate_twap(order, bars, max_bar_participation_rate=1.0)

    assert result.summary.arrival_price == pytest.approx(10.0)
    assert result.summary.average_fill_price == pytest.approx(11.0)
    assert result.summary.implementation_shortfall_bps == pytest.approx(
        expected_shortfall_bps
    )


def test_arrival_price_comes_from_first_executable_window_bar() -> None:
    bars = _make_bars(
        volume=[100, 100, 100],
        vwap=[99.0, None, 12.0],
        open_=[99.0, 10.0, 12.0],
        high=[99.0, 14.0, 12.0],
        low=[99.0, 8.0, 12.0],
        close=[99.0, 12.0, 12.0],
    )
    order = _make_order(side="buy", quantity=2, start_time=time(9, 31), end_time=time(9, 33))

    result = simulate_twap(order, bars, max_bar_participation_rate=1.0)

    assert result.summary.start_timestamp.strftime("%H:%M") == "09:31"
    assert result.summary.arrival_price == pytest.approx(11.0)


def test_session_vwap_uses_full_trade_date_and_bar_price_fallback() -> None:
    bars = _make_bars(
        volume=[100, 300, 600],
        vwap=[10.0, None, 20.0],
        open_=[10.0, 10.0, 20.0],
        high=[10.0, 14.0, 20.0],
        low=[10.0, 10.0, 20.0],
        close=[10.0, 14.0, 20.0],
    )
    order = _make_order(side="buy", quantity=2, start_time=time(9, 31), end_time=time(9, 33))

    result = simulate_twap(order, bars, max_bar_participation_rate=1.0)

    expected_session_vwap = ((100 * 10.0) + (300 * 12.0) + (600 * 20.0)) / 1000
    assert result.summary.session_vwap == pytest.approx(expected_session_vwap)


def test_partial_fill_metrics_remain_consistent_when_cap_binds() -> None:
    bars = _make_bars(
        volume=[100, 100],
        vwap=[10.0, 20.0],
    )
    order = _make_order(side="buy", quantity=100, start_time=time(9, 30), end_time=time(9, 32))

    result = simulate_twap(order, bars, max_bar_participation_rate=0.10)

    assert result.execution_log["filled_qty"].tolist() == [10, 10]
    assert result.summary.filled_qty == 20
    assert result.summary.unfilled_qty == 80
    assert result.summary.average_fill_price == pytest.approx(15.0)
    assert result.summary.filled_notional == pytest.approx(300.0)
    assert result.summary.implementation_shortfall_bps == pytest.approx(5000.0)
    assert result.summary.vwap_slippage_bps == pytest.approx(0.0)


def test_zero_fill_metrics_are_explicitly_empty_where_execution_price_is_missing() -> None:
    bars = _make_bars(
        volume=[100, 100],
        vwap=[10.0, 20.0],
    )
    order = _make_order(side="buy", quantity=10, start_time=time(9, 30), end_time=time(9, 32))

    result = simulate_twap(order, bars, max_bar_participation_rate=0.0)

    assert result.summary.filled_qty == 0
    assert result.summary.unfilled_qty == 10
    assert result.summary.average_fill_price is None
    assert result.summary.filled_notional == pytest.approx(0.0)
    assert result.summary.implementation_shortfall_bps is None
    assert result.summary.vwap_slippage_bps is None
    assert result.summary.arrival_price == pytest.approx(10.0)
    assert result.summary.session_vwap == pytest.approx(15.0)
    assert result.summary.completion_rate == pytest.approx(0.0)
    assert result.summary.realized_participation == pytest.approx(0.0)


def _make_order(
    side: str,
    quantity: int,
    start_time: time,
    end_time: time,
) -> ParentOrder:
    return ParentOrder(
        symbol="AAPL",
        side=side,
        quantity=quantity,
        trade_date=date(2026, 3, 16),
        start_time=start_time,
        end_time=end_time,
    )


def _make_bars(
    volume: list[int],
    vwap: list[float | None],
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
) -> pd.DataFrame:
    n_bars = len(volume)
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
