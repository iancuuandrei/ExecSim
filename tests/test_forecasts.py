from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from execsim.forecasting import HistoricalProfileForecaster, VolumeForecast


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session, volumes in (
        ("2026-03-12", [100, 200, 300]),
        ("2026-03-13", [200, 200, 400]),
        ("2026-03-16", [9_999, 1, 1]),
    ):
        for timestamp, volume in zip(
            pd.date_range(f"{session} 09:30", periods=3, freq="min", tz="America/New_York"),
            volumes,
            strict=True,
        ):
            rows.append({"symbol": "AAPL", "timestamp": timestamp, "volume": volume})
    return pd.DataFrame(rows)


def test_historical_profile_uses_only_sessions_preceding_target() -> None:
    target_buckets = tuple(
        pd.date_range("2026-03-16 09:30", periods=3, freq="min", tz="America/New_York")
    )
    forecaster = HistoricalProfileForecaster(_history(), estimator="mean")
    forecast = forecaster.forecast(
        symbol="AAPL",
        session_date=date(2026, 3, 16),
        generated_at=target_buckets[0],
        bucket_timestamps=target_buckets,
    )

    assert forecast.training_data_cutoff == date(2026, 3, 13)
    assert forecast.normalized_shares == pytest.approx(
        ((1 / 6 + 1 / 4) / 2, (2 / 6 + 1 / 4) / 2, (3 / 6 + 2 / 4) / 2)
    )
    assert sum(forecast.normalized_shares) == pytest.approx(1.0)


def test_forecast_rejects_future_generated_buckets_and_target_day_cutoff() -> None:
    buckets = tuple(pd.date_range("2026-03-16 09:30", periods=2, freq="min", tz="America/New_York"))
    with pytest.raises(ValueError, match="cannot precede"):
        VolumeForecast(
            symbol="AAPL",
            session_date=date(2026, 3, 16),
            generated_at=buckets[1],
            first_forecast_bucket=buckets[0],
            bucket_timestamps=buckets,
            expected_volumes=(1.0, 1.0),
            normalized_shares=(0.5, 0.5),
            expected_remaining_volume=2.0,
            forecaster_id="bad",
            feature_schema_version="v1",
            training_data_cutoff=date(2026, 3, 16),
            data_manifest_hash="x",
        )
