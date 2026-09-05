"""Resumable Alpaca SIP acquisition planning with explicit network authorization."""

from __future__ import annotations

import calendar
import hashlib
import os
from dataclasses import asdict
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from execsim.data.paper.manifests import write_json_atomic
from execsim.data.paper.schemas import (
    AcquisitionChunk,
    AcquisitionReceipt,
    PaperDataConfig,
    ProviderResponse,
)
from execsim.data.paper.validation import expected_xnys_minutes, validate_exact_xnys_session


class ChunkFetcher(Protocol):
    """Provider boundary that returns the immutable raw response bytes."""

    def __call__(self, chunk: AcquisitionChunk) -> ProviderResponse: ...


def create_alpaca_sip_fetcher() -> ChunkFetcher:
    """Create the licensed SIP-only provider adapter from environment credentials."""
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY.")
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'data' or 'paper' extra for Alpaca acquisition.") from exc

    client = StockHistoricalDataClient(api_key, api_secret)

    def fetch(chunk: AcquisitionChunk) -> ProviderResponse:
        request = StockBarsRequest(
            symbol_or_symbols=chunk.symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(chunk.start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(chunk.end, datetime.max.time(), tzinfo=UTC),
            adjustment=Adjustment.RAW,
            feed=DataFeed.SIP,
        )
        response = cast(Any, client.get_stock_bars(request))
        frame = response.df.reset_index()
        frame["instrument_id"] = chunk.instrument_id
        frame["symbol"] = chunk.symbol
        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        return ProviderResponse(buffer.getvalue(), len(frame))

    return fetch


def monthly_chunks(
    instrument_id: str, symbol: str, start: date, end: date
) -> tuple[AcquisitionChunk, ...]:
    """Create stable inclusive monthly request chunks."""
    if start > end:
        raise ValueError("Acquisition start must not follow end.")
    chunks = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = date(
            cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]
        )
        chunks.append(
            AcquisitionChunk(instrument_id, symbol.upper(), max(start, cursor), min(end, month_end))
        )
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return tuple(chunks)


def authorize_acquisition(config: PaperDataConfig, *, cli_enabled: bool) -> None:
    """Require both configuration and command-line network authorization."""
    if not config.allow_network or not cli_enabled:
        raise PermissionError(
            "Paper data acquisition is disabled. Set allow_network=true and pass "
            "--enable-network after separately authorizing the download."
        )
    if config.feed != "sip":
        raise ValueError("The paper corpus requires Alpaca SIP; IEX fallback is prohibited.")


def write_failure_receipt(
    path: Path, chunk: AcquisitionChunk, error: Exception, *, paper_config_hash: str = ""
) -> Path:
    """Persist a failed attempt without creating a false complete marker."""
    receipt = AcquisitionReceipt(
        chunk.identity,
        "failed",
        chunk.start.isoformat(),
        chunk.end.isoformat(),
        chunk.feed,
        chunk.adjustment,
        datetime.now(UTC).isoformat(),
        None,
        0,
        instrument_id=chunk.instrument_id,
        symbol=chunk.symbol,
        error=f"{type(error).__name__}: {error}",
        paper_config_hash=paper_config_hash,
    )
    return write_json_atomic(path, asdict(receipt))


def acquire_chunk(
    chunk: AcquisitionChunk,
    *,
    output_directory: Path,
    fetch: ChunkFetcher,
    config: PaperDataConfig,
    cli_enabled: bool,
    max_attempts: int = 3,
) -> AcquisitionReceipt:
    """Fetch one idempotent chunk with bounded retries and atomic completion metadata."""
    authorize_acquisition(config, cli_enabled=cli_enabled)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / f"{chunk.identity}.response"
    receipt_path = output_directory / f"{chunk.identity}.json"
    if raw_path.is_file() and receipt_path.is_file():
        from execsim.data.paper.manifests import read_json

        existing = AcquisitionReceipt(**read_json(receipt_path))
        if (
            existing.status == "complete"
            and existing.chunk_identity == chunk.identity
            and existing.instrument_id == chunk.instrument_id
            and existing.symbol == chunk.symbol
            and existing.requested_start == chunk.start.isoformat()
            and existing.requested_end == chunk.end.isoformat()
            and existing.feed == chunk.feed
            and existing.adjustment == chunk.adjustment
            and existing.paper_config_hash == config.paper_config_hash
            and existing.response_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        ):
            observed, expected = _validate_provider_response(
                chunk, raw_path.read_bytes(), existing.row_count
            )
            if observed != existing.observed_sessions or expected != existing.expected_sessions:
                raise ValueError(
                    f"Existing acquisition coverage metadata is inconsistent: {chunk.identity}"
                )
            return existing
        raise ValueError(f"Existing acquisition chunk is inconsistent: {chunk.identity}")
    last_error: Exception | None = None
    temporary = raw_path.with_suffix(raw_path.suffix + ".part")
    for _ in range(max_attempts):
        try:
            response = fetch(chunk)
            observed_sessions, expected_sessions = _validate_provider_response(
                chunk, response.content, response.row_count
            )
            temporary.write_bytes(response.content)
            os.replace(temporary, raw_path)
            receipt = AcquisitionReceipt(
                chunk.identity,
                "complete",
                chunk.start.isoformat(),
                chunk.end.isoformat(),
                chunk.feed,
                chunk.adjustment,
                datetime.now(UTC).isoformat(),
                hashlib.sha256(response.content).hexdigest(),
                response.row_count,
                instrument_id=chunk.instrument_id,
                symbol=chunk.symbol,
                observed_sessions=observed_sessions,
                expected_sessions=expected_sessions,
                provider_request_id=response.request_id,
                paper_config_hash=config.paper_config_hash,
            )
            write_json_atomic(receipt_path, asdict(receipt))
            return receipt
        except Exception as exc:  # provider implementations expose heterogeneous failures
            last_error = exc
            temporary.unlink(missing_ok=True)
    if last_error is None:  # pragma: no cover - max_attempts validation makes this unreachable
        raise RuntimeError("Acquisition failed without an exception.")
    write_failure_receipt(
        receipt_path, chunk, last_error, paper_config_hash=config.paper_config_hash
    )
    raise RuntimeError(
        f"Acquisition failed after {max_attempts} attempts: {chunk.identity}"
    ) from last_error


def _validate_provider_response(
    chunk: AcquisitionChunk, content: bytes, declared_rows: int
) -> tuple[int, int]:
    """Validate provider Parquet schema, identity, interval, and session coverage."""
    try:
        frame = pd.read_parquet(BytesIO(content))
    except Exception as exc:
        raise ValueError("Provider response is not readable Parquet.") from exc
    if len(frame) != declared_rows or declared_rows <= 0:
        raise ValueError("Provider response row count is zero or does not match metadata.")
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
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Provider response schema is incomplete: {sorted(missing)}")
    if set(frame["instrument_id"].astype(str)) != {chunk.instrument_id}:
        raise ValueError("Provider response instrument identity does not match request.")
    if set(frame["symbol"].astype(str).str.upper()) != {chunk.symbol.upper()}:
        raise ValueError("Provider response symbol does not match request interval.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is None:
        raise ValueError("Provider response timestamps must be timezone-aware.")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    if local_dates.min() < chunk.start or local_dates.max() > chunk.end:
        raise ValueError("Provider response contains rows outside the exact request interval.")
    observed = 0
    for _, session in frame.groupby(local_dates, sort=True):
        expected = expected_xnys_minutes(pd.to_datetime(session["timestamp"]).iloc[0].date())
        if len(expected) == 390 and not validate_exact_xnys_session(session):
            observed += 1
    calendar = __import__("exchange_calendars").get_calendar("XNYS")
    expected_sessions = len(calendar.sessions_in_range(chunk.start, chunk.end))
    if expected_sessions <= 0 or observed <= 0:
        raise ValueError("Provider response has no validated regular-session coverage.")
    return observed, expected_sessions
