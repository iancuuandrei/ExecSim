"""Pre-request acquisition sizing and durable empirical-program planning receipts."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from execsim.data.paper.acquisition import monthly_chunks
from execsim.data.paper.manifests import stable_hash, write_json_atomic
from execsim.data.paper.schemas import InstrumentSymbolInterval


def build_acquisition_plan(
    *,
    snapshot: pd.DataFrame,
    intervals: tuple[InstrumentSymbolInterval, ...],
    formation_start: date,
    formation_end: date,
    target_start: date,
    target_end: date,
    target_universe_size: int,
    spy_instrument_id: str,
    output_directory: Path,
    paper_config_hash: str,
) -> dict[str, object]:
    """Write request, row, storage, and source bounds before the first Alpaca call."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    formation_sessions = len(calendar.sessions_in_range(formation_start, formation_end))
    target_sessions = len(calendar.sessions_in_range(target_start, target_end))
    formation_ids = tuple(
        dict.fromkeys((*snapshot["instrument_id"].astype(str), spy_instrument_id))
    )
    formation_requests = _request_count(formation_ids, intervals, formation_start, formation_end)
    target_instruments = target_universe_size + 1
    target_months = len(monthly_chunks("planned", "PLANNED", target_start, target_end))
    target_requests_upper = target_instruments * target_months
    formation_rows = len(formation_ids) * formation_sessions * 390
    target_rows = target_instruments * target_sessions * 390
    total_rows = formation_rows + target_rows
    raw_bytes_per_row = 80
    processed_bytes_per_row = 120
    expected_bytes = total_rows * (raw_bytes_per_row + processed_bytes_per_row)
    free_bytes = shutil.disk_usage(output_directory.resolve().anchor).free
    stable = {
        "schema_version": "paper-acquisition-plan-v1",
        "paper_config_hash": paper_config_hash,
        "provider": "alpaca",
        "feed": "sip",
        "frequency": "1min",
        "adjustment": "raw",
        "regular_session_minutes": 390,
        "formation": {
            "start": formation_start.isoformat(),
            "end": formation_end.isoformat(),
            "candidate_instruments_including_spy": len(formation_ids),
            "expected_sessions_per_instrument": formation_sessions,
            "expected_minute_rows_upper": formation_rows,
            "monthly_requests": formation_requests,
        },
        "target": {
            "start": target_start.isoformat(),
            "end": target_end.isoformat(),
            "instruments_including_spy": target_instruments,
            "expected_sessions_per_instrument": target_sessions,
            "expected_minute_rows_upper": target_rows,
            "monthly_requests_upper": target_requests_upper,
        },
        "sources": {
            "formation_membership": "pinned constituent-history revision",
            "ticker_history": "pinned constituent-history revision with fail-closed ambiguity",
            "stable_identity": "SEC CIK plus formation share-class symbol",
            "corporate_actions": "Alpaca v1 corporate-actions complete records",
        },
        "storage_estimate": {
            "assumed_raw_compressed_bytes_per_row": raw_bytes_per_row,
            "assumed_processed_bytes_per_row": processed_bytes_per_row,
            "expected_total_bytes": expected_bytes,
            "free_local_bytes_before_acquisition": free_bytes,
            "headroom_ratio": free_bytes / expected_bytes,
        },
        "provider_probe": {
            "symbol": "AAPL",
            "date": formation_start.isoformat(),
            "required_rows": 390,
            "must_verify": [
                "credentials",
                "sip_feed",
                "historical_period",
                "pagination",
                "raw_adjustment",
                "exact_xnys_grid",
            ],
        },
    }
    payload = {**stable, "plan_sha256": stable_hash(stable)}
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_directory / "acquisition-plan.json", payload)
    markdown = _plan_markdown(payload)
    (output_directory / "ACQUISITION_PLAN.md").write_text(markdown, encoding="utf-8")
    return payload


def _request_count(
    instrument_ids: tuple[str, ...],
    intervals: tuple[InstrumentSymbolInterval, ...],
    start: date,
    end: date,
) -> int:
    return sum(
        len(
            monthly_chunks(
                interval.instrument_id,
                interval.symbol,
                max(start, interval.start),
                min(end, interval.end),
            )
        )
        for interval in intervals
        if interval.instrument_id in instrument_ids
        and max(start, interval.start) <= min(end, interval.end)
    )


def _plan_markdown(payload: dict[str, object]) -> str:
    formation = payload["formation"]
    target = payload["target"]
    storage = payload["storage_estimate"]
    assert isinstance(formation, dict) and isinstance(target, dict) and isinstance(storage, dict)
    gib = 1024**3
    return (
        "# Sparse-JEPA v1 acquisition plan\n\n"
        "This plan was frozen before the first Alpaca request. Counts are upper bounds until "
        "session exclusions and the formation universe are observed.\n\n"
        "## Requests and rows\n\n"
        f"- Formation: {formation['candidate_instruments_including_spy']} instruments including "
        f"SPY, {formation['expected_sessions_per_instrument']} expected sessions per instrument, "
        f"{formation['monthly_requests']} monthly requests, and "
        f"{formation['expected_minute_rows_upper']:,} minute rows at full coverage.\n"
        f"- Target: {target['instruments_including_spy']} instruments including SPY, "
        f"{target['expected_sessions_per_instrument']} expected sessions per instrument, at most "
        f"{target['monthly_requests_upper']} monthly requests, and "
        f"{target['expected_minute_rows_upper']:,} minute rows at full coverage.\n\n"
        "## Storage bound\n\n"
        f"The planning assumption is {storage['assumed_raw_compressed_bytes_per_row']} compressed "
        f"raw bytes and {storage['assumed_processed_bytes_per_row']} processed bytes per row. "
        f"The resulting bound is {storage['expected_total_bytes'] / gib:.2f} GiB; local free space "
        f"was {storage['free_local_bytes_before_acquisition'] / gib:.2f} GiB.\n\n"
        "## First request gate\n\n"
        "The first provider call is one AAPL formation-date session. Bulk acquisition starts only "
        "if credentials, SIP identity, raw adjustment, historical access, pagination behavior, and "
        "the exact 390-minute XNYS grid validate.\n"
    )
