"""Fail-closed validation for regular-session paper bars."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from execsim.data.paper.schemas import PAPER_BAR_COLUMNS


def classify_session(*, observed_minutes: int, calendar_early_close: bool) -> str:
    """Classify a session for primary, labeled robustness, or exclusion use."""
    if observed_minutes == 390 and not calendar_early_close:
        return "primary_regular"
    if calendar_early_close and 0 < observed_minutes < 390:
        return "robustness_early_close"
    return "excluded_incomplete_or_ambiguous"


def validate_paper_bars(bars: pd.DataFrame, *, expected_minutes: int = 390) -> tuple[str, ...]:
    """Return durable exclusion reasons; missing bars are never interpreted as zero."""
    errors: list[str] = []
    missing = set(PAPER_BAR_COLUMNS).difference(bars.columns)
    if missing:
        return (f"missing columns: {sorted(missing)}",)
    timestamps = pd.to_datetime(bars["timestamp"], errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is None:
        errors.append("timestamps must be valid and timezone-aware")
    if timestamps.duplicated().any():
        errors.append("duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        errors.append("timestamps must be strictly ordered")
    if len(bars) != expected_minutes:
        errors.append(
            f"regular session requires {expected_minutes} observed minutes, got {len(bars)}"
        )
    numeric = bars.loc[:, ["open", "high", "low", "close", "volume", "trade_count", "vwap"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append("bar values must be finite")
    if (numeric[["open", "high", "low", "close", "vwap"]] <= 0).any().any():
        errors.append("prices must be positive")
    if (numeric[["volume", "trade_count"]] < 0).any().any():
        errors.append("volume and trade_count must be non-negative")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any() or (
        numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    ).any():
        errors.append("OHLC high/low invariants are violated")
    return tuple(errors)


def expected_xnys_minutes(session_date: object) -> pd.DatetimeIndex:
    """Return the authoritative regular-session minute grid in New York time."""
    return _cached_xnys_minutes(pd.Timestamp(session_date).date())


@lru_cache(maxsize=2_048)
def _cached_xnys_minutes(session_date: date) -> pd.DatetimeIndex:
    """Cache immutable calendar grids across corpus-wide instrument validation."""
    try:
        import exchange_calendars as exchange_calendars
    except ImportError as exc:  # pragma: no cover - optional paper dependency
        raise RuntimeError("Install the 'paper' extra for XNYS calendar validation.") from exc
    day = pd.Timestamp(session_date)
    try:
        minutes = exchange_calendars.get_calendar("XNYS").session_minutes(day)
    except Exception as exc:
        raise ValueError(f"{day.date()} is not a valid XNYS session.") from exc
    return minutes.tz_convert("America/New_York")


def validate_exact_xnys_session(bars: pd.DataFrame) -> tuple[str, ...]:
    """Validate one instrument against the exact full XNYS regular-minute grid."""
    errors = list(validate_paper_bars(bars))
    if "timestamp" not in bars:
        return tuple(errors)
    timestamps = pd.to_datetime(bars["timestamp"], errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is None:
        return tuple(dict.fromkeys((*errors, "timestamps must be valid and timezone-aware")))
    timezone_name = str(timestamps.dt.tz)
    if timezone_name != "America/New_York":
        errors.append(f"timezone must be exactly America/New_York, got {timezone_name}")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date.unique()
    if len(local_dates) != 1:
        errors.append("paper session must contain exactly one local session date")
        return tuple(dict.fromkeys(errors))
    if "instrument_id" in bars and bars["instrument_id"].nunique(dropna=False) != 1:
        errors.append("paper session must contain exactly one instrument")
    if "symbol" in bars and bars["symbol"].nunique(dropna=False) != 1:
        errors.append("paper session must contain exactly one symbol")
    expected = expected_xnys_minutes(local_dates[0])
    if len(expected) != 390:
        errors.append("early-close or nonstandard XNYS session is excluded from primary data")
    actual = pd.DatetimeIndex(timestamps)
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        errors.append(
            f"timestamps do not match exact XNYS grid (missing={len(missing)}, extra={len(extra)})"
        )
    return tuple(dict.fromkeys(errors))
