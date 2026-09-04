from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from execsim.forecasting.models import VolumeForecast

ProfileEstimator = Literal["mean", "median", "ewma", "previous"]


@dataclass(slots=True)
class HistoricalProfileForecaster:
    """Point-in-time volume curve estimated only from preceding complete windows."""

    historical_bars: pd.DataFrame
    estimator: ProfileEstimator = "mean"
    lookback_sessions: int | None = 20
    ewma_alpha: float = 0.25
    pooled: bool = False
    feature_schema_version: str = "volume-profile-v1"
    data_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        required = {"symbol", "timestamp", "volume"}
        missing = required.difference(self.historical_bars.columns)
        if missing:
            raise ValueError(f"Historical bars missing required columns: {sorted(missing)}")
        if self.estimator not in {"mean", "median", "ewma", "previous"}:
            raise ValueError(f"Unknown profile estimator: {self.estimator}")
        if self.lookback_sessions is not None and self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive or None.")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1].")
        bars = self.historical_bars.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        if bars["timestamp"].dt.tz is None:
            raise ValueError("Historical profile timestamps must be timezone-aware.")
        volumes = pd.to_numeric(bars["volume"], errors="coerce")
        if volumes.isna().any() or (volumes < 0).any():
            raise ValueError("Historical profile volumes must be finite and non-negative.")
        bars["volume"] = volumes.astype(float)
        self.historical_bars = bars
        if self.data_manifest_hash is None:
            digest_input = bars.loc[:, ["symbol", "timestamp", "volume"]].to_csv(index=False)
            self.data_manifest_hash = hashlib.sha256(digest_input.encode()).hexdigest()

    @property
    def provider_id(self) -> str:
        scope = "pooled" if self.pooled else "symbol"
        return f"historical-{self.estimator}-{scope}-v1"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        del observations  # deterministic V1 profile does not adapt its scale intraday
        timestamps = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        if not timestamps:
            raise ValueError("At least one future bucket is required.")
        bars = self.historical_bars
        prior = bars.loc[bars["timestamp"].dt.date < session_date].copy()
        if not self.pooled:
            prior = prior.loc[prior["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if prior.empty:
            raise ValueError(f"No prior sessions are available for {symbol} before {session_date}.")

        expected_times = [timestamp.strftime("%H:%M") for timestamp in timestamps]
        prior["session_key"] = (
            prior["symbol"].astype(str).str.upper() + "|" + prior["timestamp"].dt.date.astype(str)
        )
        prior["bucket_time"] = prior["timestamp"].dt.strftime("%H:%M")
        prior = prior.loc[prior["bucket_time"].isin(expected_times)].copy()
        counts = prior.groupby("session_key")["bucket_time"].nunique()
        complete_keys = counts[counts == len(set(expected_times))].index
        prior = prior.loc[prior["session_key"].isin(complete_keys)].copy()
        if prior.empty:
            raise ValueError(
                "No preceding sessions contain the complete requested forecast window."
            )

        session_order = (
            prior.groupby("session_key")["timestamp"].min().sort_values(kind="stable").index
        )
        if self.lookback_sessions is not None:
            session_order = session_order[-self.lookback_sessions :]
        prior = prior.loc[prior["session_key"].isin(session_order)].copy()
        pivot = prior.pivot_table(
            index="session_key",
            columns="bucket_time",
            values="volume",
            aggfunc="sum",
        ).reindex(index=session_order, columns=expected_times)
        if pivot.isna().any().any():
            raise ValueError("Historical profile construction found missing buckets.")
        totals = pivot.sum(axis=1)
        positive = totals > 0
        pivot = pivot.loc[positive]
        totals = totals.loc[positive]
        if pivot.empty:
            raise ValueError("Historical profile requires at least one positive-volume session.")
        shapes = pivot.div(totals, axis=0)

        if self.estimator == "mean":
            raw_shape = shapes.mean(axis=0).to_numpy(float)
            expected_total = float(totals.mean())
        elif self.estimator == "median":
            raw_shape = shapes.median(axis=0).to_numpy(float)
            expected_total = float(totals.median())
        elif self.estimator == "previous":
            raw_shape = shapes.iloc[-1].to_numpy(float)
            expected_total = float(totals.iloc[-1])
        else:
            count = len(shapes)
            weights = (1.0 - self.ewma_alpha) ** np.arange(count - 1, -1, -1)
            weights /= weights.sum()
            raw_shape = np.average(shapes.to_numpy(float), axis=0, weights=weights)
            expected_total = float(np.average(totals.to_numpy(float), weights=weights))

        raw_shape = np.maximum(raw_shape, 0.0)
        shape = raw_shape / raw_shape.sum()
        expected = expected_total * shape
        cutoff = prior["timestamp"].dt.date.max()
        warnings: list[str] = []
        if len(shapes) < 5:
            warnings.append("fewer_than_five_complete_prior_sessions")
        return VolumeForecast(
            symbol=symbol.upper(),
            session_date=session_date,
            generated_at=pd.Timestamp(generated_at),
            first_forecast_bucket=timestamps[0],
            bucket_timestamps=timestamps,
            expected_volumes=tuple(float(value) for value in expected),
            normalized_shares=tuple(float(value) for value in shape),
            expected_remaining_volume=float(expected.sum()),
            forecaster_id=self.provider_id,
            feature_schema_version=self.feature_schema_version,
            training_data_cutoff=cutoff,
            data_manifest_hash=str(self.data_manifest_hash),
            warnings=tuple(warnings),
        )
