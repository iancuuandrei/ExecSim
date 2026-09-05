"""Daily-resolution formation universe for the sparse-JEPA v2 protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256, stable_hash, write_json_atomic
from execsim.data.paper.resolution_quality import validate_daily_observation

DAILY_COMPLETENESS_MINIMUM = 0.95


@dataclass(frozen=True, slots=True)
class V2UniverseMember:
    """One frozen member selected from direct daily formation observations."""

    rank: int
    instrument_id: str
    symbol: str
    median_daily_price: float
    daily_completeness: float
    median_daily_share_volume: float
    median_daily_dollar_volume: float
    liquidity_group: int


def build_daily_formation_candidates(
    snapshot: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    expected_session_dates: tuple[object, ...],
    identity_source_hash: str,
) -> pd.DataFrame:
    """Rebuild v2 candidates from direct daily bars without using v1 statistics."""
    expected_dates = {pd.Timestamp(value).date() for value in expected_session_dates}
    if not expected_dates:
        raise ValueError("V2 formation requires expected XNYS daily sessions.")
    required_snapshot = {"instrument_id", "symbol", "security_type"}
    if missing := required_snapshot.difference(snapshot.columns):
        raise ValueError(f"V2 formation snapshot missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for constituent in snapshot.sort_values("instrument_id", kind="stable").itertuples(index=False):
        instrument_id = str(constituent.instrument_id)
        symbol = str(constituent.symbol).upper()
        selected = daily_bars.loc[
            daily_bars["instrument_id"].astype(str) == instrument_id
        ].sort_values("timestamp", kind="stable")
        valid_rows: list[pd.Series] = []
        invalid_reasons: list[str] = []
        observed_dates: set[date] = set()
        for _, row in selected.iterrows():
            errors = validate_daily_observation(row, expected_dates=expected_dates)
            if errors:
                invalid_reasons.extend(errors)
                continue
            session_date = pd.Timestamp(row["timestamp"]).tz_convert("America/New_York").date()
            if session_date in observed_dates:
                invalid_reasons.append("duplicate_daily_observation")
                continue
            observed_dates.add(session_date)
            valid_rows.append(row)
        valid = pd.DataFrame(valid_rows, columns=daily_bars.columns)
        completeness = len(valid) / len(expected_dates)
        median_price = _median(valid, "close")
        median_volume = _median(valid, "volume")
        median_dollar = (
            float(
                np.median(
                    pd.to_numeric(valid["vwap"]).to_numpy(dtype=float)
                    * pd.to_numeric(valid["volume"]).to_numpy(dtype=float)
                )
            )
            if len(valid)
            else 0.0
        )
        reasons = sorted(set(invalid_reasons))
        if str(constituent.security_type) != "ordinary_common_stock":
            reasons.append("not_ordinary_common_stock")
        if median_price < 5.0:
            reasons.append("median_daily_price_below_5")
        if completeness < DAILY_COMPLETENESS_MINIMUM:
            reasons.append("daily_completeness_below_95_percent")
        if median_dollar <= 0:
            reasons.append("nonpositive_median_daily_dollar_volume")
        records.append(
            {
                "instrument_id": instrument_id,
                "formation_symbol": symbol,
                "in_sp500_on_formation_date": True,
                "security_type": str(constituent.security_type),
                "expected_daily_sessions": len(expected_dates),
                "observed_valid_daily_sessions": len(valid),
                "daily_completeness": completeness,
                "median_daily_price": median_price,
                "median_daily_share_volume": median_volume,
                "median_daily_dollar_volume": median_dollar,
                "first_valid_formation_date": (
                    min(observed_dates).isoformat() if observed_dates else ""
                ),
                "last_valid_formation_date": (
                    max(observed_dates).isoformat() if observed_dates else ""
                ),
                "identity_source_hash": identity_source_hash,
                "formation_data_hash": _frame_hash(valid),
                "exclusion_reasons": json.dumps(sorted(set(reasons)), separators=(",", ":")),
            }
        )
    return pd.DataFrame(records).sort_values("instrument_id", kind="stable").reset_index(drop=True)


def select_v2_universe(
    candidates: pd.DataFrame, *, size: int = 100
) -> tuple[V2UniverseMember, ...]:
    """Select the fixed v2 top 100 under the predeclared daily-quality rule."""
    eligible = candidates.loc[
        (candidates["security_type"] == "ordinary_common_stock")
        & candidates["in_sp500_on_formation_date"].astype(bool)
        & (candidates["median_daily_price"] >= 5.0)
        & (candidates["daily_completeness"] >= DAILY_COMPLETENESS_MINIMUM)
        & (candidates["median_daily_dollar_volume"] > 0)
        & candidates["instrument_id"].astype(str).str.len().gt(0)
    ].copy()
    eligible = eligible.sort_values(
        ["median_daily_dollar_volume", "instrument_id"],
        ascending=[False, True],
        kind="stable",
    )
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} daily-quality universe members; {size} required.")
    selected = eligible.head(size)
    return tuple(
        V2UniverseMember(
            rank=rank,
            instrument_id=str(row.instrument_id),
            symbol=str(row.formation_symbol).upper(),
            median_daily_price=float(row.median_daily_price),
            daily_completeness=float(row.daily_completeness),
            median_daily_share_volume=float(row.median_daily_share_volume),
            median_daily_dollar_volume=float(row.median_daily_dollar_volume),
            liquidity_group=min(5, (rank - 1) * 5 // size + 1),
        )
        for rank, row in enumerate(selected.itertuples(index=False), start=1)
    )


def write_v2_universe_manifest(
    members: tuple[V2UniverseMember, ...],
    *,
    output: Path,
    source_hashes: tuple[str, ...],
    symbol_history: tuple[dict[str, str], ...],
    paper_config_hash: str,
) -> Path:
    """Freeze a version-separated daily-quality universe manifest."""
    if len(members) != 100 or tuple(item.rank for item in members) != tuple(range(1, 101)):
        raise ValueError("Completed v2 universe must contain ranks 1 through 100.")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "rank": item.rank,
            "instrument_id": item.instrument_id,
            "formation_symbol": item.symbol,
            "liquidity_group": item.liquidity_group,
            "median_daily_price": item.median_daily_price,
            "median_daily_share_volume": item.median_daily_share_volume,
            "median_daily_dollar_volume": item.median_daily_dollar_volume,
            "daily_completeness": item.daily_completeness,
        }
        for item in members
    ]
    stable = {
        "schema_version": "paper-universe-v2",
        "protocol_id": "sparse-jepa-v2",
        "formation_membership_date": "2021-01-04",
        "ranking_available_after": "2021-12-31",
        "formation_period": ["2021-01-04", "2021-12-31"],
        "daily_completeness_minimum": DAILY_COMPLETENESS_MINIMUM,
        "target_size": 100,
        "source_hashes": list(source_hashes),
        "members": rows,
        "symbol_history": list(symbol_history),
        "paper_config_hash": paper_config_hash,
    }
    return write_json_atomic(
        output,
        {
            **stable,
            "status": "complete",
            "universe_id": "universe-v2-" + stable_hash(stable)[:16],
        },
    )


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return float(np.median(pd.to_numeric(frame[column]).to_numpy(dtype=float)))


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    values = frame.sort_values("timestamp", kind="stable").to_json(
        orient="records", date_format="iso", double_precision=15
    )
    return hashlib.sha256(values.encode()).hexdigest()


def v2_candidate_artifact_hashes(candidate_path: Path, daily_path: Path) -> dict[str, str]:
    """Return explicit supporting-artifact identities for quality reports."""
    return {
        "candidate_table_sha256": file_sha256(candidate_path),
        "daily_corpus_sha256": file_sha256(daily_path),
    }
