from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from execsim.forecasting.models import VolumeForecast


@dataclass(slots=True)
class RealizedVolumeOracleForecaster:
    """Evaluation-only provider that intentionally reads realized future volume."""

    realized_bars: pd.DataFrame
    provider_id: str = "oracle-realized-volume-evaluation-only-v1"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        del observations
        timestamps = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        bars = self.realized_bars.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        bars = bars.loc[
            (bars["symbol"].astype(str).str.upper() == symbol.upper())
            & (bars["timestamp"].dt.date == session_date)
            & bars["timestamp"].isin(timestamps)
        ].sort_values("timestamp", kind="stable")
        if tuple(bars["timestamp"]) != timestamps:
            raise ValueError("Oracle forecast requires every realized future bucket.")
        volumes = bars["volume"].to_numpy(dtype=float)
        if np.any(volumes < 0) or not np.isfinite(volumes).all():
            raise ValueError("Oracle volumes must be finite and non-negative.")
        total = float(volumes.sum())
        shares = volumes / total if total else np.zeros_like(volumes)
        return VolumeForecast(
            symbol=symbol.upper(),
            session_date=session_date,
            generated_at=generated_at,
            first_forecast_bucket=timestamps[0],
            bucket_timestamps=timestamps,
            expected_volumes=tuple(float(value) for value in volumes),
            normalized_shares=tuple(float(value) for value in shares),
            expected_remaining_volume=total,
            forecaster_id=self.provider_id,
            feature_schema_version="oracle-realized-v1",
            training_data_cutoff=None,
            data_manifest_hash="evaluation-only-target-session",
            warnings=("EVALUATION_ONLY: uses realized future target-session volume",),
        )
