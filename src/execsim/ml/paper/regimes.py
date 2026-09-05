"""Training-only thresholds and deterministic paper regime labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Store train-fold 90th-percentile thresholds."""

    volume_surprise: float
    realized_volatility: float
    shape_error: float
    quantile: float = 0.9


@dataclass(frozen=True, slots=True)
class IndependentProbeThresholds:
    """Train-only thresholds for volume, volatility, shape, and SPY state probes."""

    volume_surprise: float
    realized_volatility: float
    shape_error: float
    absolute_spy_return: float
    quantile: float = 0.9


@dataclass(frozen=True, slots=True)
class HistoricalBaselineRegimeThreshold:
    """Freeze one exploratory threshold for an identically computed baseline statistic."""

    statistic: str
    threshold: float
    quantile: float = 0.9


@dataclass(frozen=True, slots=True)
class UnusualSessionThresholds:
    """TRAIN-only thresholds for the one predeclared unusual-session composite."""

    volume_surprise: float
    realized_volatility: float
    historical_baseline_curve_error: float
    quantile: float = 0.9


def fit_unusual_session_thresholds(training: pd.DataFrame) -> UnusualSessionThresholds:
    """Fit each component of the locked OR composite on TRAIN only."""
    columns = (
        "volume_surprise",
        "realized_volatility",
        "historical_baseline_curve_error",
    )
    if missing := set(columns).difference(training.columns):
        raise ValueError(f"Unusual-session training rows missing columns: {sorted(missing)}")
    values = training.loc[:, columns].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Unusual-session thresholds require finite TRAIN rows.")
    return UnusualSessionThresholds(*map(float, np.quantile(values, 0.9, axis=0)))


def label_unusual_sessions(
    rows: pd.DataFrame, thresholds: UnusualSessionThresholds
) -> pd.DataFrame:
    """Apply the exact same historical-baseline OR statistic to any partition."""
    result = rows.copy()
    result["unusual_session"] = (
        (result["volume_surprise"] > thresholds.volume_surprise)
        | (result["realized_volatility"] > thresholds.realized_volatility)
        | (result["historical_baseline_curve_error"] > thresholds.historical_baseline_curve_error)
    )
    result["regime"] = np.where(result["unusual_session"], "unusual", "ordinary")
    return result


def fit_historical_baseline_regime(
    training: pd.DataFrame, *, statistic: str = "historical_baseline_unusual_score"
) -> HistoricalBaselineRegimeThreshold:
    """Fit one TRAIN-only unusual-market composite threshold."""
    if statistic not in training:
        raise ValueError(f"Regime rows require the declared statistic: {statistic}")
    values = training[statistic].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Historical-baseline regime statistic must be finite and non-empty.")
    return HistoricalBaselineRegimeThreshold(statistic, float(np.quantile(values, 0.9)))


def label_historical_baseline_regime(
    rows: pd.DataFrame, threshold: HistoricalBaselineRegimeThreshold
) -> pd.DataFrame:
    """Apply the exact TRAIN statistic definition to held-out rows."""
    if threshold.statistic not in rows:
        raise ValueError(f"Held-out regime rows require the same statistic: {threshold.statistic}")
    result = rows.copy()
    values = result[threshold.statistic].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Held-out historical-baseline statistic must be finite.")
    result["unusual_market"] = values >= threshold.threshold
    result["ordinary_market"] = ~result["unusual_market"]
    return result


def fit_regime_thresholds(training: pd.DataFrame) -> RegimeThresholds:
    """Fit the three locked thresholds from training rows only."""
    columns = ("volume_surprise", "realized_volatility", "shape_error")
    missing = set(columns).difference(training.columns)
    if missing:
        raise ValueError(f"Regime training rows missing columns: {sorted(missing)}")
    values = training.loc[:, columns].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Regime training inputs must be non-empty and finite.")
    thresholds = np.quantile(values, 0.9, axis=0)
    return RegimeThresholds(*map(float, thresholds))


def label_regimes(rows: pd.DataFrame, thresholds: RegimeThresholds) -> pd.DataFrame:
    """Add independent unusual-state flags and an ordinary complement."""
    result = rows.copy()
    result["high_volume_surprise"] = result["volume_surprise"] >= thresholds.volume_surprise
    result["high_volatility"] = result["realized_volatility"] >= thresholds.realized_volatility
    result["abnormal_shape"] = result["shape_error"] >= thresholds.shape_error
    result["ordinary"] = ~result[["high_volume_surprise", "high_volatility", "abnormal_shape"]].any(
        axis=1
    )
    return result


def fit_independent_probe_thresholds(training: pd.DataFrame) -> IndependentProbeThresholds:
    """Fit independent state labels from TRAIN rows without embedding inputs."""
    required = {"volume_surprise", "realized_volatility", "shape_error", "spy_return"}
    if missing := required.difference(training.columns):
        raise ValueError(f"Independent probe training rows missing columns: {sorted(missing)}")
    values = training.loc[:, ["volume_surprise", "realized_volatility", "shape_error"]]
    if training.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Independent probe thresholds require finite TRAIN rows.")
    return IndependentProbeThresholds(
        float(training["volume_surprise"].quantile(0.9)),
        float(training["realized_volatility"].quantile(0.9)),
        float(training["shape_error"].quantile(0.9)),
        float(training["spy_return"].abs().quantile(0.9)),
    )


def label_independent_probe_states(
    rows: pd.DataFrame, thresholds: IndependentProbeThresholds
) -> pd.DataFrame:
    """Label externally defined probe targets; never derive states from embeddings."""
    result = rows.copy()
    result["high_volume_surprise"] = result["volume_surprise"] >= thresholds.volume_surprise
    result["high_volatility"] = result["realized_volatility"] >= thresholds.realized_volatility
    result["abnormal_shape"] = result["shape_error"] >= thresholds.shape_error
    result["large_spy_move"] = result["spy_return"].abs() >= thresholds.absolute_spy_return
    result["ordinary"] = ~result[
        ["high_volume_surprise", "high_volatility", "abnormal_shape", "large_spy_move"]
    ].any(axis=1)
    return result
