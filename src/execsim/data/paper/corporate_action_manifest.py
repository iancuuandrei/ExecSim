"""Sourced point-in-time corporate-action metadata for sequence preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from execsim.data.paper.manifests import file_sha256, stable_hash, write_json_atomic


def ingest_corporate_actions(path: Path) -> pd.DataFrame:
    """Load split factors with stable identity, effective date, and availability time."""
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"instrument_id", "effective_date", "factor", "available_at", "source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Corporate-action source missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    if (
        frame[["effective_date", "available_at", "factor"]].isna().any().any()
        or (frame["factor"] <= 0).any()
        or frame["source"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError("Corporate-action rows contain invalid dates, factors, or provenance.")
    if frame.duplicated(["instrument_id", "effective_date"]).any():
        raise ValueError("Corporate-action rows duplicate instrument/effective-date identity.")
    return frame.sort_values(["instrument_id", "effective_date"], kind="stable")


def write_corporate_action_manifest(
    source: Path, actions: pd.DataFrame, output: Path, *, paper_config_hash: str = ""
) -> Path:
    """Persist the sourced corporate-action identity without inventing missing aliases."""
    stable = {
        "schema_version": "paper-corporate-actions-v1",
        "source_sha256": file_sha256(source),
        "row_count": len(actions),
        "instruments": sorted(actions["instrument_id"].astype(str).unique()),
        "paper_config_hash": paper_config_hash,
    }
    return write_json_atomic(output, {**stable, "manifest_hash": stable_hash(stable)})
