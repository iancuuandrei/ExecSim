from __future__ import annotations

from datetime import date, time
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execsim.orders import ParentOrder
from execsim.strategies.twap import TwapStrategy


def test_twap_schedule_sums_to_requested_quantity_before_caps() -> None:
    bars = _make_bars(n_bars=4)
    order = ParentOrder(
        symbol="AAPL",
        side="buy",
        quantity=10,
        trade_date=date(2026, 3, 16),
        start_time=time(9, 30),
        end_time=time(9, 34),
    )

    schedule = TwapStrategy().generate_schedule(order, bars)

    assert schedule["scheduled_qty"].tolist() == [3, 3, 2, 2]
    assert int(schedule["scheduled_qty"].sum()) == order.quantity


def test_twap_schedule_handles_quantity_smaller_than_bar_count() -> None:
    bars = _make_bars(n_bars=5)
    order = ParentOrder(
        symbol="AAPL",
        side="sell",
        quantity=2,
        trade_date=date(2026, 3, 16),
        start_time=time(9, 30),
        end_time=time(9, 35),
    )

    schedule = TwapStrategy().generate_schedule(order, bars)

    assert schedule["scheduled_qty"].tolist() == [1, 1, 0, 0, 0]
    assert int(schedule["scheduled_qty"].sum()) == order.quantity


def _make_bars(n_bars: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-03-16 09:30",
                periods=n_bars,
                freq="min",
                tz="America/New_York",
            )
        }
    )
