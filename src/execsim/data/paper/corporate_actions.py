"""Point-in-time split adjustment for ML features only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_in_time_split_factor(
    actions: pd.DataFrame,
    *,
    instrument_id: str,
    observation_at: pd.Timestamp,
    market_information_as_of: pd.Timestamp,
) -> float:
    """Resolve only actions both effective and known at the declared market instant."""
    if observation_at.tzinfo is None or market_information_as_of.tzinfo is None:
        raise ValueError("Corporate-action observation and information times must be aware.")
    required = {"instrument_id", "effective_date", "factor", "available_at"}
    if missing := required.difference(actions.columns):
        raise ValueError(f"Corporate-action frame missing columns: {sorted(missing)}")
    known = pd.to_datetime(actions["available_at"], utc=True)
    effective = pd.to_datetime(actions["effective_date"], utc=True)
    selected = actions.loc[
        (actions["instrument_id"].astype(str) == instrument_id)
        & (known <= market_information_as_of.tz_convert("UTC"))
        & (effective <= observation_at.tz_convert("UTC"))
    ]
    factor = float(pd.to_numeric(selected["factor"], errors="raise").prod())
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Point-in-time split factor must be finite and positive.")
    return factor


def apply_point_in_time_split_adjustment(
    bars: pd.DataFrame,
    factor: pd.Series,
    *,
    factor_available_at: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Restate price and volume while preserving per-bar dollar notional."""
    if len(bars) != len(factor) or factor.isna().any() or (factor <= 0).any():
        raise ValueError("Split factors must align, be finite, and be strictly positive.")
    if (factor_available_at is None) != (as_of is None):
        raise ValueError("factor_available_at and as_of must be supplied together.")
    if factor_available_at is not None and as_of is not None:
        available = pd.to_datetime(factor_available_at, errors="coerce")
        if available.isna().any() or available.dt.tz is None or as_of.tzinfo is None:
            raise ValueError("Split-factor availability and as_of must be timezone-aware.")
        if (available > as_of).any():
            raise ValueError("Split adjustment cannot use a factor known after as_of.")
    adjusted = bars.copy()
    values = factor.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Split factors must be finite.")
    for column in ("open", "high", "low", "close", "vwap"):
        adjusted[column] = pd.to_numeric(adjusted[column]) / values
    adjusted["volume"] = pd.to_numeric(adjusted["volume"]) * values
    original = pd.to_numeric(bars["vwap"]).to_numpy() * pd.to_numeric(bars["volume"]).to_numpy()
    restated = adjusted["vwap"].to_numpy() * adjusted["volume"].to_numpy()
    if not np.allclose(original, restated, rtol=1e-12, atol=1e-8):
        raise RuntimeError("Split adjustment violated dollar-notional invariance.")
    return adjusted
