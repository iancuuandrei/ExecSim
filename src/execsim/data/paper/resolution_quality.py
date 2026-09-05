"""Resolution-aware data quality for sparse-JEPA v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time
from typing import Literal, overload

import numpy as np
import pandas as pd

from execsim.data.paper.schemas import PAPER_BAR_COLUMNS
from execsim.data.paper.validation import expected_xnys_minutes

TOKEN_MINUTES = 15
TOKEN_COUNT = 26
MINIMUM_OBSERVED_BARS_PER_TOKEN = 2
TCA_START = time(10, 30)
TCA_END = time(15, 30)


@dataclass(frozen=True, slots=True)
class SessionResolutionQuality:
    """Keep daily, minute, token, and replay quality as independent facts."""

    instrument_id: str
    symbol: str
    session_date: str
    daily_valid: bool
    minute_exact_full_session: bool
    token_valid_full_session: bool
    tca_window_exact: bool
    early_close: bool
    provider_gap_count: int
    observed_minute_count: int
    valid_token_count: int
    invalid_token_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable manifest row."""
        return asdict(self)


def validate_daily_observation(row: pd.Series, *, expected_dates: set[date]) -> tuple[str, ...]:
    """Validate one direct provider daily bar against the v2 formation contract."""
    errors: list[str] = []
    missing = set(PAPER_BAR_COLUMNS).difference(row.index)
    if missing:
        return (f"missing columns: {sorted(missing)}",)
    timestamp = pd.Timestamp(row["timestamp"])
    if timestamp.tzinfo is None:
        errors.append("daily timestamp must be timezone-aware")
    else:
        local_date = timestamp.tz_convert("America/New_York").date()
        if local_date not in expected_dates:
            errors.append("daily timestamp is not an expected XNYS session")
    numeric = pd.to_numeric(
        row[["open", "high", "low", "close", "volume", "trade_count", "vwap"]],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("daily values must be finite")
        return tuple(errors)
    open_, high, low, close, volume, trade_count, vwap = numeric
    if min(open_, high, low, close, vwap) <= 0:
        errors.append("daily prices must be positive")
    if volume <= 0 or trade_count <= 0:
        errors.append("daily volume and trade count must be positive")
    if high < max(open_, close, low) or low > min(open_, close, high):
        errors.append("daily OHLC high/low invariants are violated")
    if not str(row["instrument_id"]).strip() or not str(row["symbol"]).strip():
        errors.append("daily stable identity and symbol are required")
    return tuple(errors)


def aggregate_observed_tokens(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed provider bars into fixed v2 tokens without fabricating minutes."""
    quality, tokens = assess_session_resolution_quality(bars, return_tokens=True)
    if not quality.token_valid_full_session or tokens is None:
        raise ValueError("Token-invalid v2 session: " + quality.invalid_token_reason)
    return tokens


@overload
def assess_session_resolution_quality(
    bars: pd.DataFrame,
    *,
    daily_valid: bool = False,
    return_tokens: Literal[False] = False,
) -> SessionResolutionQuality: ...


@overload
def assess_session_resolution_quality(
    bars: pd.DataFrame,
    *,
    daily_valid: bool = False,
    return_tokens: Literal[True],
) -> tuple[SessionResolutionQuality, pd.DataFrame | None]: ...


def assess_session_resolution_quality(
    bars: pd.DataFrame,
    *,
    daily_valid: bool = False,
    return_tokens: bool = False,
) -> SessionResolutionQuality | tuple[SessionResolutionQuality, pd.DataFrame | None]:
    """Assess one observed provider session at minute, token, and TCA resolutions."""
    required = set(PAPER_BAR_COLUMNS)
    missing = required.difference(bars.columns)
    if missing or bars.empty:
        quality = SessionResolutionQuality(
            instrument_id=_single_text(bars, "instrument_id"),
            symbol=_single_text(bars, "symbol"),
            session_date="",
            daily_valid=daily_valid,
            minute_exact_full_session=False,
            token_valid_full_session=False,
            tca_window_exact=False,
            early_close=False,
            provider_gap_count=0,
            observed_minute_count=len(bars),
            valid_token_count=0,
            invalid_token_reason=(
                f"missing_columns:{','.join(sorted(missing))}" if missing else "empty_session"
            ),
        )
        return (quality, None) if return_tokens else quality
    if not return_tokens:
        return _assess_quality_only(bars, daily_valid=daily_valid)

    ordered = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
    reasons: list[str] = []
    if timestamps.isna().any() or timestamps.dt.tz is None:
        reasons.append("invalid_or_naive_timestamp")
        quality = _invalid_quality(ordered, daily_valid, reasons)
        return (quality, None) if return_tokens else quality
    if str(timestamps.dt.tz) != "America/New_York":
        reasons.append("wrong_timezone")
    if timestamps.duplicated().any():
        reasons.append("duplicate_timestamp")
    if not pd.DatetimeIndex(pd.to_datetime(bars["timestamp"])).is_monotonic_increasing:
        reasons.append("reordered_timestamp")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date.unique()
    if len(local_dates) != 1:
        reasons.append("multiple_session_dates")
        quality = _invalid_quality(ordered, daily_valid, reasons)
        return (quality, None) if return_tokens else quality
    if ordered["instrument_id"].nunique(dropna=False) != 1:
        reasons.append("multiple_instruments")
    if ordered["symbol"].nunique(dropna=False) != 1:
        reasons.append("multiple_symbols")
    reasons.extend(_observed_numeric_errors(ordered))

    session_date = local_dates[0]
    expected = expected_xnys_minutes(session_date)
    early_close = len(expected) != 390
    actual = pd.DatetimeIndex(timestamps)
    extra = actual.difference(expected)
    if len(extra):
        reasons.append(f"off_grid_minutes:{len(extra)}")
    minute_exact = not reasons and actual.equals(expected)
    provider_gap_count = max(len(expected.difference(actual)), 0)

    tokens: pd.DataFrame | None = None
    token_reasons: list[str] = list(reasons)
    valid_token_count = 0
    if early_close:
        token_reasons.append("early_close_not_primary_26_token_session")
    elif not reasons:
        tokens, per_token_reasons = _aggregate_standard_session(ordered, session_date)
        token_reasons.extend(per_token_reasons)
        valid_token_count = TOKEN_COUNT - len(per_token_reasons)
    tca_expected = pd.date_range(
        pd.Timestamp.combine(session_date, TCA_START).tz_localize("America/New_York"),
        periods=300,
        freq="min",
    )
    tca_actual = actual[(actual.time >= TCA_START) & (actual.time < TCA_END)]
    tca_exact = not reasons and tca_actual.equals(tca_expected)
    token_valid = not token_reasons and tokens is not None and len(tokens) == TOKEN_COUNT
    quality = SessionResolutionQuality(
        instrument_id=str(ordered["instrument_id"].iloc[0]),
        symbol=str(ordered["symbol"].iloc[0]).upper(),
        session_date=session_date.isoformat(),
        daily_valid=daily_valid,
        minute_exact_full_session=minute_exact,
        token_valid_full_session=token_valid,
        tca_window_exact=tca_exact,
        early_close=early_close,
        provider_gap_count=provider_gap_count,
        observed_minute_count=len(ordered),
        valid_token_count=valid_token_count,
        invalid_token_reason=";".join(token_reasons),
    )
    return (quality, tokens if token_valid else None) if return_tokens else quality


def _assess_quality_only(bars: pd.DataFrame, *, daily_valid: bool) -> SessionResolutionQuality:
    """Calculate quality identities without materializing token aggregates."""
    ordered = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
    reasons: list[str] = []
    if timestamps.isna().any() or timestamps.dt.tz is None:
        return _invalid_quality(ordered, daily_valid, ["invalid_or_naive_timestamp"])
    if str(timestamps.dt.tz) != "America/New_York":
        reasons.append("wrong_timezone")
    if timestamps.duplicated().any():
        reasons.append("duplicate_timestamp")
    original = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"]))
    if not original.is_monotonic_increasing:
        reasons.append("reordered_timestamp")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date.unique()
    if len(local_dates) != 1:
        return _invalid_quality(ordered, daily_valid, [*reasons, "multiple_session_dates"])
    if ordered["instrument_id"].nunique(dropna=False) != 1:
        reasons.append("multiple_instruments")
    if ordered["symbol"].nunique(dropna=False) != 1:
        reasons.append("multiple_symbols")
    reasons.extend(_observed_numeric_errors(ordered))

    session_date = local_dates[0]
    expected = expected_xnys_minutes(session_date)
    early_close = len(expected) != 390
    actual = pd.DatetimeIndex(timestamps)
    actual_ns = actual.as_unit("ns").asi8
    expected_ns = expected.as_unit("ns").asi8
    on_grid = np.isin(actual_ns, expected_ns, assume_unique=False)
    off_grid_count = int((~on_grid).sum())
    if off_grid_count:
        reasons.append(f"off_grid_minutes:{off_grid_count}")
    minute_exact = not reasons and np.array_equal(actual_ns, expected_ns)
    provider_gap_count = int((~np.isin(expected_ns, actual_ns, assume_unique=False)).sum())

    token_reasons = list(reasons)
    valid_token_count = 0
    if early_close:
        token_reasons.append("early_close_not_primary_26_token_session")
    elif not reasons:
        offsets = actual.hour * 60 + actual.minute - (9 * 60 + 30)
        buckets = np.asarray(offsets // TOKEN_MINUTES, dtype=int)
        counts = np.bincount(buckets, minlength=TOKEN_COUNT)
        volume = pd.to_numeric(ordered["volume"]).to_numpy(dtype=float)
        volume_sums = np.bincount(buckets, weights=volume, minlength=TOKEN_COUNT)
        valid = (counts[:TOKEN_COUNT] >= MINIMUM_OBSERVED_BARS_PER_TOKEN) & (
            volume_sums[:TOKEN_COUNT] > 0
        )
        valid_token_count = int(valid.sum())
        for bucket_id in np.flatnonzero(counts[:TOKEN_COUNT] < MINIMUM_OBSERVED_BARS_PER_TOKEN):
            token_reasons.append(f"token_{bucket_id:02d}_observed_bars_lt_2")
        for bucket_id in np.flatnonzero(
            (counts[:TOKEN_COUNT] >= MINIMUM_OBSERVED_BARS_PER_TOKEN)
            & (volume_sums[:TOKEN_COUNT] <= 0)
        ):
            token_reasons.append(f"token_{bucket_id:02d}_nonpositive_volume")
    tca_start = pd.Timestamp.combine(session_date, TCA_START).tz_localize("America/New_York")
    tca_expected_ns = pd.date_range(tca_start, periods=300, freq="min").as_unit("ns").asi8
    tca_mask = (actual.time >= TCA_START) & (actual.time < TCA_END)
    tca_exact = not reasons and np.array_equal(actual_ns[tca_mask], tca_expected_ns)
    token_valid = not token_reasons and valid_token_count == TOKEN_COUNT
    return SessionResolutionQuality(
        instrument_id=str(ordered["instrument_id"].iloc[0]),
        symbol=str(ordered["symbol"].iloc[0]).upper(),
        session_date=session_date.isoformat(),
        daily_valid=daily_valid,
        minute_exact_full_session=minute_exact,
        token_valid_full_session=token_valid,
        tca_window_exact=tca_exact,
        early_close=early_close,
        provider_gap_count=provider_gap_count,
        observed_minute_count=len(ordered),
        valid_token_count=valid_token_count,
        invalid_token_reason=";".join(token_reasons),
    )


def _aggregate_standard_session(
    bars: pd.DataFrame, session_date: date
) -> tuple[pd.DataFrame, list[str]]:
    timestamps = pd.to_datetime(bars["timestamp"])
    minute_offset = timestamps.dt.hour * 60 + timestamps.dt.minute - (9 * 60 + 30)
    bucket_ids = (minute_offset // TOKEN_MINUTES).astype(int)
    rows: list[dict[str, object]] = []
    reasons: list[str] = []
    for bucket_id in range(TOKEN_COUNT):
        bucket = bars.loc[bucket_ids == bucket_id]
        if len(bucket) < MINIMUM_OBSERVED_BARS_PER_TOKEN:
            reasons.append(f"token_{bucket_id:02d}_observed_bars_lt_2")
            continue
        closes = pd.to_numeric(bucket["close"]).to_numpy(dtype=float)
        volume_values = pd.to_numeric(bucket["volume"]).to_numpy(dtype=float)
        volume = float(volume_values.sum())
        if volume <= 0:
            reasons.append(f"token_{bucket_id:02d}_nonpositive_volume")
            continue
        vwap_values = pd.to_numeric(bucket["vwap"]).to_numpy(dtype=float)
        log_returns = np.diff(np.log(closes))
        end = pd.Timestamp.combine(session_date, time(9, 30)).tz_localize(
            "America/New_York"
        ) + pd.Timedelta(minutes=(bucket_id + 1) * TOKEN_MINUTES)
        rows.append(
            {
                "timestamp": end,
                "open": float(pd.to_numeric(bucket["open"]).iloc[0]),
                "high": float(pd.to_numeric(bucket["high"]).max()),
                "low": float(pd.to_numeric(bucket["low"]).min()),
                "close": float(closes[-1]),
                "volume": volume,
                "trade_count": float(pd.to_numeric(bucket["trade_count"]).sum()),
                "vwap": float(np.dot(vwap_values, volume_values) / volume),
                "realized_volatility": float(np.sqrt(np.square(log_returns).sum())),
                "observed_bar_count": len(bucket),
            }
        )
    return pd.DataFrame(rows), reasons


def _observed_numeric_errors(bars: pd.DataFrame) -> list[str]:
    numeric = bars.loc[:, ["open", "high", "low", "close", "volume", "trade_count", "vwap"]].apply(
        pd.to_numeric, errors="coerce"
    )
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        return ["nonfinite_bar_values"]
    if (numeric[["open", "high", "low", "close", "vwap"]] <= 0).any().any():
        errors.append("nonpositive_price")
    if (numeric[["volume", "trade_count"]] < 0).any().any():
        errors.append("negative_volume_or_trade_count")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any() or (
        numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    ).any():
        errors.append("invalid_ohlc")
    return errors


def _invalid_quality(
    bars: pd.DataFrame, daily_valid: bool, reasons: list[str]
) -> SessionResolutionQuality:
    return SessionResolutionQuality(
        instrument_id=_single_text(bars, "instrument_id"),
        symbol=_single_text(bars, "symbol"),
        session_date="",
        daily_valid=daily_valid,
        minute_exact_full_session=False,
        token_valid_full_session=False,
        tca_window_exact=False,
        early_close=False,
        provider_gap_count=0,
        observed_minute_count=len(bars),
        valid_token_count=0,
        invalid_token_reason=";".join(reasons),
    )


def _single_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame or frame.empty:
        return ""
    values = frame[column].drop_duplicates()
    return str(values.iloc[0]) if len(values) == 1 else ""
