"""Constituent-snapshot ingestion and formation-period universe statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256, write_json_atomic
from execsim.data.paper.validation import validate_exact_xnys_session

FORMATION_DATE = pd.Timestamp("2021-01-04").date()


@dataclass(frozen=True, slots=True)
class FormationExclusion:
    """Record why one snapshot instrument did not enter the eligible candidate set."""

    instrument_id: str
    symbol: str
    reasons: tuple[str, ...]


def ingest_constituent_snapshot(path: Path) -> pd.DataFrame:
    """Load a sourced point-in-time constituent snapshot with stable identities."""
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {
        "instrument_id",
        "symbol",
        "security_type",
        "effective_date",
        "source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Constituent snapshot missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
    if dates.isna().any() or set(dates) != {FORMATION_DATE}:
        raise ValueError("Constituent snapshot must be effective exactly on 2021-01-04.")
    if frame["instrument_id"].astype(str).str.len().eq(0).any():
        raise ValueError("Constituent snapshot contains an empty stable instrument identity.")
    if frame["instrument_id"].duplicated().any():
        raise ValueError("Constituent snapshot contains duplicate stable instrument identities.")
    if frame["source"].astype(str).str.len().eq(0).any():
        raise ValueError("Constituent snapshot requires source provenance.")
    return frame.sort_values("instrument_id", kind="stable").reset_index(drop=True)


def build_formation_candidates(
    snapshot: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    expected_session_count: int,
) -> tuple[pd.DataFrame, tuple[FormationExclusion, ...]]:
    """Compute price, completeness, and dollar-volume statistics plus exclusions."""
    if expected_session_count <= 0:
        raise ValueError("Formation statistics require a positive expected session count.")
    required = {
        "instrument_id",
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Formation bar corpus missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    exclusions: list[FormationExclusion] = []
    for constituent in snapshot.itertuples(index=False):
        instrument_id = str(constituent.instrument_id)
        symbol = str(constituent.symbol).upper()
        instrument = bars.loc[bars["instrument_id"].astype(str) == instrument_id].copy()
        valid_sessions: list[pd.DataFrame] = []
        if not instrument.empty:
            local_dates = (
                pd.to_datetime(instrument["timestamp"]).dt.tz_convert("America/New_York").dt.date
            )
            for _, session in instrument.groupby(local_dates, sort=True):
                if not validate_exact_xnys_session(session):
                    valid_sessions.append(session)
        daily_dollar = np.asarray(
            [
                float((pd.to_numeric(session["vwap"]) * pd.to_numeric(session["volume"])).sum())
                for session in valid_sessions
            ],
            dtype=float,
        )
        prices = (
            pd.to_numeric(pd.concat(valid_sessions)["close"]).to_numpy(dtype=float)
            if valid_sessions
            else np.asarray([], dtype=float)
        )
        completeness = len(valid_sessions) / expected_session_count
        median_price = float(np.median(prices)) if len(prices) else 0.0
        median_dollar = float(np.median(daily_dollar)) if len(daily_dollar) else 0.0
        reasons = []
        if str(constituent.security_type) != "ordinary_common_stock":
            reasons.append("not_ordinary_common_stock")
        if median_price < 5.0:
            reasons.append("median_price_below_5")
        if completeness < 0.95:
            reasons.append("formation_session_completeness_below_95_percent")
        if median_dollar <= 0:
            reasons.append("nonpositive_median_daily_dollar_volume")
        records.append(
            {
                "instrument_id": instrument_id,
                "symbol": symbol,
                "security_type": str(constituent.security_type),
                "in_sp500_on_formation_date": True,
                "median_price": median_price,
                "session_completeness": completeness,
                "median_daily_dollar_volume": median_dollar,
            }
        )
        if reasons:
            exclusions.append(FormationExclusion(instrument_id, symbol, tuple(reasons)))
    return pd.DataFrame(records).sort_values("instrument_id", kind="stable"), tuple(exclusions)


def write_formation_receipts(
    path: Path,
    *,
    snapshot_path: Path,
    candidates: pd.DataFrame,
    exclusions: tuple[FormationExclusion, ...],
    paper_config_hash: str = "",
) -> Path:
    """Persist auditable eligibility and exclusion inputs."""
    payload = {
        "schema_version": "paper-formation-receipts-v1",
        "snapshot_sha256": file_sha256(snapshot_path),
        "candidate_count": len(candidates),
        "eligible_count": len(candidates) - len(exclusions),
        "exclusions": [asdict(item) for item in exclusions],
        "paper_config_hash": paper_config_hash,
    }
    return write_json_atomic(path, payload)
