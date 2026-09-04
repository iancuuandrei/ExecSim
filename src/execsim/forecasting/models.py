from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class VolumeForecast:
    symbol: str
    session_date: date
    generated_at: pd.Timestamp
    first_forecast_bucket: pd.Timestamp
    bucket_timestamps: tuple[pd.Timestamp, ...]
    expected_volumes: tuple[float, ...]
    normalized_shares: tuple[float, ...]
    expected_remaining_volume: float
    forecaster_id: str
    feature_schema_version: str
    training_data_cutoff: date | None
    data_manifest_hash: str
    lower_quantiles: tuple[float, ...] | None = None
    median_quantiles: tuple[float, ...] | None = None
    upper_quantiles: tuple[float, ...] | None = None
    warnings: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lengths = {
            len(self.bucket_timestamps),
            len(self.expected_volumes),
            len(self.normalized_shares),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError(
                "Forecast timestamps, volumes, and shares must be equal and non-empty."
            )
        if self.generated_at.tzinfo is None or self.first_forecast_bucket.tzinfo is None:
            raise ValueError("Forecast timestamps must be timezone-aware.")
        if self.first_forecast_bucket != self.bucket_timestamps[0]:
            raise ValueError("first_forecast_bucket must equal the first bucket timestamp.")
        if any(timestamp.tzinfo is None for timestamp in self.bucket_timestamps):
            raise ValueError("All forecast bucket timestamps must be timezone-aware.")
        if any(timestamp < self.generated_at for timestamp in self.bucket_timestamps):
            raise ValueError("Forecast buckets cannot precede generated_at.")
        if any(timestamp.date() != self.session_date for timestamp in self.bucket_timestamps):
            raise ValueError("Forecast buckets must belong to session_date.")
        if any(
            not math.isfinite(value) or value < 0
            for value in (*self.expected_volumes, *self.normalized_shares)
        ):
            raise ValueError("Forecast volumes and shares must be finite and non-negative.")
        total = sum(self.expected_volumes)
        if not math.isfinite(self.expected_remaining_volume) or self.expected_remaining_volume < 0:
            raise ValueError("expected_remaining_volume must be finite and non-negative.")
        if not math.isclose(total, self.expected_remaining_volume, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Expected bucket volumes must sum to expected_remaining_volume.")
        if total > 0 and not math.isclose(sum(self.normalized_shares), 1.0, abs_tol=1e-8):
            raise ValueError("Positive-volume forecast shares must sum to one.")
        if total == 0 and any(self.normalized_shares):
            raise ValueError("Zero-volume forecasts must have all-zero shares.")
        if self.training_data_cutoff is not None and self.training_data_cutoff >= self.session_date:
            raise ValueError("training_data_cutoff must precede the forecast session.")
        for quantiles in (self.lower_quantiles, self.median_quantiles, self.upper_quantiles):
            if quantiles is not None and len(quantiles) != len(self.bucket_timestamps):
                raise ValueError("Forecast quantiles must match the forecast horizon.")


class VolumeForecastProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast: ...
