"""Pinned public constituent history and deterministic point-in-time identity inputs."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from execsim.data.paper.manifests import file_sha256, write_json_atomic

COMPONENTS_REVISION = "ed4cf46e5ec5bb02e709aa08ee8a3a218d1b7d19"
COMPONENTS_URL = (
    "https://raw.githubusercontent.com/lawcal/sp500-components-history/"
    f"{COMPONENTS_REVISION}/data/components_history.csv"
)
COMPONENTS_SHA256 = "0c248c94e708f33a6235688c47aadfccc0d7779c545fd65c6c8b698dcf964c1b"
COMPONENTS_LICENSE = (
    f"https://github.com/lawcal/sp500-components-history/blob/{COMPONENTS_REVISION}/LICENSE"
)


def acquire_constituent_identity_sources(
    *,
    formation_date: date,
    target_end: date,
    snapshot_output: Path,
    ticker_history_output: Path,
    receipt_output: Path,
    spy_instrument_id: str,
    content: bytes | None = None,
) -> dict[str, object]:
    """Materialize a formation snapshot and sourced symbol intervals from a pinned revision."""
    if content is None:
        request = Request(COMPONENTS_URL, headers={"User-Agent": "ExecSim/1 data-research"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != COMPONENTS_SHA256:
        raise ValueError("Pinned constituent-history bytes do not match the frozen checksum.")
    history = pd.read_csv(BytesIO(content), dtype={"cik": "string"})
    required = {
        "symbol",
        "cik",
        "name",
        "sector",
        "date_added",
        "date_removed",
        "created_at",
    }
    if missing := required.difference(history.columns):
        raise ValueError(f"Constituent-history source is incomplete: {sorted(missing)}")
    history = history.copy()
    for column in ("date_added", "date_removed", "created_at"):
        history[column] = pd.to_datetime(
            history[column].astype("string").str.rstrip("*"), errors="coerce"
        )
    if history[["date_added", "created_at"]].isna().any().any():
        raise ValueError("Constituent-history source has unparseable required dates.")
    instant = pd.Timestamp(formation_date)
    snapshot_rows = history.loc[
        (history["date_added"] <= instant)
        & (history["created_at"] <= instant)
        & (history["date_removed"].isna() | (history["date_removed"] > instant))
    ].copy()
    if len(snapshot_rows) < 500 or snapshot_rows["symbol"].duplicated().any():
        raise ValueError("Formation snapshot is implausible or contains duplicate symbols.")
    source = f"{COMPONENTS_URL}#sha256={COMPONENTS_SHA256}"
    snapshot_rows["symbol"] = snapshot_rows["symbol"].astype(str).str.upper()
    snapshot_rows["instrument_id"] = [
        _instrument_id(str(cik), str(symbol))
        for cik, symbol in zip(snapshot_rows["cik"], snapshot_rows["symbol"], strict=True)
    ]
    snapshot = snapshot_rows.assign(
        security_type="ordinary_common_stock",
        effective_date=formation_date.isoformat(),
        source=source,
    ).loc[:, ["instrument_id", "symbol", "security_type", "effective_date", "source"]]

    intervals: list[dict[str, object]] = []
    cik_counts = snapshot_rows.groupby("cik")["symbol"].nunique()
    for row in snapshot_rows.itertuples(index=False):
        cik = str(row.cik)
        initial_symbol = str(row.symbol)
        instrument_id = _instrument_id(cik, initial_symbol)
        transitions = [(formation_date, initial_symbol)]
        if int(cik_counts.loc[cik]) == 1:
            future = history.loc[
                (history["cik"].astype(str) == cik)
                & (history["created_at"] > instant)
                & (history["created_at"] <= pd.Timestamp(target_end)),
                ["created_at", "symbol"],
            ].drop_duplicates()
            transitions.extend(
                (value.created_at.date(), str(value.symbol).upper())
                for value in future.sort_values(["created_at", "symbol"]).itertuples(index=False)
            )
        transitions = list(dict.fromkeys(transitions))
        for index, (start, symbol) in enumerate(transitions):
            end = (
                transitions[index + 1][0] - timedelta(days=1)
                if index + 1 < len(transitions)
                else target_end
            )
            if start <= end:
                intervals.append(
                    {
                        "instrument_id": instrument_id,
                        "symbol": symbol,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "source": source,
                    }
                )
    intervals.append(
        {
            "instrument_id": spy_instrument_id,
            "symbol": "SPY",
            "start": formation_date.isoformat(),
            "end": target_end.isoformat(),
            "source": "Alpaca SIP symbol identity verified by acquisition probe",
        }
    )
    ticker_history = pd.DataFrame(intervals).sort_values(
        ["instrument_id", "start", "symbol"], kind="stable"
    )
    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    ticker_history_output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(snapshot_output, index=False)
    ticker_history.to_parquet(ticker_history_output, index=False)
    receipt = {
        "schema_version": "paper-formation-source-v1",
        "source_url": COMPONENTS_URL,
        "source_revision": COMPONENTS_REVISION,
        "source_sha256": digest,
        "source_license": COMPONENTS_LICENSE,
        "formation_date": formation_date.isoformat(),
        "snapshot_rows": len(snapshot),
        "ticker_intervals": len(ticker_history),
        "snapshot_sha256": file_sha256(snapshot_output),
        "ticker_history_sha256": file_sha256(ticker_history_output),
        "classification_note": (
            "The source enumerates S&P 500 company share classes; retained rows are classified "
            "as ordinary common stock for the paper eligibility field."
        ),
        "identity_note": (
            "Stable IDs combine SEC CIK and the formation share-class symbol. Later symbols are "
            "linked only for CIKs with one formation share class; ambiguous multi-class CIKs are "
            "not inferred."
        ),
    }
    write_json_atomic(receipt_output, receipt)
    return receipt


def _instrument_id(cik: str, formation_symbol: str) -> str:
    normalized_cik = cik.zfill(10)
    normalized_symbol = formation_symbol.upper().replace(".", "-")
    return f"sec-cik-{normalized_cik}-{normalized_symbol}"
