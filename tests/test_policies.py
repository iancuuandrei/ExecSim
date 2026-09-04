from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from execsim.forecasting import VolumeForecast
from execsim.orders import ParentOrder
from execsim.policies import (
    DecisionContext,
    ExecutionConstraints,
    HistoricalVwapPolicy,
    PovPolicy,
    TwapPolicy,
)


def _order() -> ParentOrder:
    return ParentOrder("AAPL", "buy", 10, date(2026, 3, 16), time(9, 30), time(9, 33))


def _context(forecast: VolumeForecast | None = None) -> DecisionContext:
    timestamps = tuple(
        pd.date_range("2026-03-16 09:30", periods=3, freq="min", tz="America/New_York")
    )
    return DecisionContext(
        current_timestamp=timestamps[0],
        decision_timing="bucket_start",
        remaining_inventory=10,
        elapsed_buckets=0,
        remaining_buckets=3,
        observations=pd.DataFrame(columns=["timestamp"]),
        future_timestamps=timestamps,
        forecast=forecast,
        constraints=ExecutionConstraints(0.2, 0.2),
    )


def _forecast() -> VolumeForecast:
    timestamps = tuple(
        pd.date_range("2026-03-16 09:30", periods=3, freq="min", tz="America/New_York")
    )
    return VolumeForecast(
        symbol="AAPL",
        session_date=date(2026, 3, 16),
        generated_at=timestamps[0],
        first_forecast_bucket=timestamps[0],
        bucket_timestamps=timestamps,
        expected_volumes=(10.0, 20.0, 70.0),
        normalized_shares=(0.1, 0.2, 0.7),
        expected_remaining_volume=100.0,
        forecaster_id="fixture",
        feature_schema_version="v1",
        training_data_cutoff=date(2026, 3, 13),
        data_manifest_hash="fixture",
    )


def test_twap_and_vwap_reconcile_integer_quantities() -> None:
    twap = TwapPolicy().create_plan(_order(), _context())
    vwap = HistoricalVwapPolicy().create_plan(_order(), _context(_forecast()))

    assert twap.quantities == (4, 3, 3)
    assert vwap.quantities == (1, 2, 7)
    assert sum(vwap.quantities) == 10


def test_pov_is_current_volume_driven_and_inventory_bounded() -> None:
    decision = PovPolicy(0.15).decide(_context(), observable_current_volume=40)

    assert decision.planned_quantity == 6
    assert "materializes" in str(decision.trace)


def test_decision_context_rejects_current_or_future_observations() -> None:
    bad = pd.DataFrame({"timestamp": [pd.Timestamp("2026-03-16 09:30", tz="America/New_York")]})
    with pytest.raises(ValueError, match="strictly in the past"):
        DecisionContext(
            current_timestamp=bad["timestamp"].iloc[0],
            decision_timing="bucket_start",
            remaining_inventory=10,
            elapsed_buckets=0,
            remaining_buckets=1,
            observations=bad,
            future_timestamps=(bad["timestamp"].iloc[0],),
            forecast=None,
            constraints=ExecutionConstraints(),
        )
