from __future__ import annotations

from datetime import date

import pandas as pd

from execsim.forecasting import VolumeForecast
from execsim.ml.forecasts import positive_volume_forecast


def prediction_rows_to_forecast(
    rows: pd.DataFrame,
    *,
    symbol: str,
    session_date: date,
    generated_at: pd.Timestamp,
    forecaster_id: str,
    feature_schema_version: str,
    training_data_cutoff: date,
    data_manifest_hash: str,
    prediction_column: str = "predicted_volume",
) -> VolumeForecast:
    required = {"target_bucket_timestamp", prediction_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Prediction rows missing columns: {sorted(missing)}")
    ordered = rows.sort_values("target_bucket_timestamp", kind="stable")
    timestamps = tuple(pd.Timestamp(value) for value in ordered["target_bucket_timestamp"])
    volumes, shares = positive_volume_forecast(ordered[prediction_column].to_numpy())
    return VolumeForecast(
        symbol=symbol.upper(),
        session_date=session_date,
        generated_at=generated_at,
        first_forecast_bucket=timestamps[0],
        bucket_timestamps=timestamps,
        expected_volumes=tuple(float(value) for value in volumes),
        normalized_shares=tuple(float(value) for value in shares),
        expected_remaining_volume=float(volumes.sum()),
        forecaster_id=forecaster_id,
        feature_schema_version=feature_schema_version,
        training_data_cutoff=training_data_cutoff,
        data_manifest_hash=data_manifest_hash,
    )
