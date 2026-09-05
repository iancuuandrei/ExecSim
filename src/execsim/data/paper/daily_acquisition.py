"""Direct Alpaca SIP daily-bar acquisition for sparse-JEPA v2 formation."""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.data.paper.resolution_quality import validate_daily_observation


def acquire_formation_daily_bars(
    snapshot: pd.DataFrame,
    *,
    formation_start: date,
    formation_end: date,
    spy_instrument_id: str,
    output_path: Path,
    receipt_path: Path,
    paper_config_hash: str,
    cli_enabled: bool,
    config_enabled: bool,
) -> dict[str, object]:
    """Acquire one paginated multi-symbol daily corpus with dual authorization."""
    if not cli_enabled or not config_enabled:
        raise PermissionError("Daily formation acquisition requires config and CLI authorization.")
    required = {"instrument_id", "symbol"}
    if missing := required.difference(snapshot.columns):
        raise ValueError(f"Formation snapshot missing daily-acquisition fields: {sorted(missing)}")
    mapping = {
        str(row.symbol).upper(): str(row.instrument_id) for row in snapshot.itertuples(index=False)
    }
    if len(mapping) != len(snapshot):
        raise ValueError("Formation symbols must map one-to-one to stable identities.")
    if "SPY" in mapping:
        raise ValueError("SPY must remain a benchmark identity, not a formation constituent.")
    mapping["SPY"] = spy_instrument_id
    request_identity = {
        "provider": "alpaca",
        "feed": "sip",
        "timeframe": "1Day",
        "adjustment": "raw",
        "asof": formation_start.isoformat(),
        "start": formation_start.isoformat(),
        "end": formation_end.isoformat(),
        "symbols": sorted(mapping),
        "paper_config_hash": paper_config_hash,
    }
    if output_path.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") == "complete"
            and receipt.get("request") == request_identity
            and receipt.get("content_sha256") == file_sha256(output_path)
        ):
            return receipt
        raise ValueError("Existing v2 daily formation artifact is incompatible.")

    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY.")
    query = {
        "symbols": ",".join(sorted(mapping)),
        "timeframe": "1Day",
        "start": formation_start.isoformat(),
        "end": formation_end.isoformat(),
        "adjustment": "raw",
        "feed": "sip",
        "asof": formation_start.isoformat(),
        "limit": "10000",
        "sort": "asc",
    }
    rows: list[dict[str, object]] = []
    pages = 0
    page_token: str | None = None
    rate_limit: dict[str, str] = {}
    while True:
        page_query = {**query, **({"page_token": page_token} if page_token else {})}
        request = Request(
            "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(page_query),
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
                rate_limit = {
                    name: value
                    for name in (
                        "X-RateLimit-Limit",
                        "X-RateLimit-Remaining",
                        "X-RateLimit-Reset",
                    )
                    if (value := response.headers.get(name)) is not None
                }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"BLOCKED: Alpaca daily formation request failed with HTTP {exc.code}: {detail}"
            ) from exc
        bars = payload.get("bars") if isinstance(payload, dict) else None
        if not isinstance(bars, dict):
            raise ValueError("Alpaca multi-symbol daily response schema is invalid.")
        unexpected = set(map(str.upper, bars)).difference(mapping)
        if unexpected:
            raise ValueError(f"Alpaca daily response contains unexpected symbols: {unexpected}")
        for symbol, observations in bars.items():
            if not isinstance(observations, list):
                raise ValueError(f"Alpaca daily observations are invalid for {symbol}.")
            for value in observations:
                if not isinstance(value, dict):
                    raise ValueError(f"Alpaca daily observation is invalid for {symbol}.")
                rows.append(
                    {
                        "instrument_id": mapping[str(symbol).upper()],
                        "symbol": str(symbol).upper(),
                        "timestamp": value.get("t"),
                        "open": value.get("o"),
                        "high": value.get("h"),
                        "low": value.get("l"),
                        "close": value.get("c"),
                        "volume": value.get("v"),
                        "trade_count": value.get("n"),
                        "vwap": value.get("vw"),
                    }
                )
        pages += 1
        page_token = payload.get("next_page_token")
        if page_token is None:
            break
        if not isinstance(page_token, str) or pages > 100:
            raise ValueError("Alpaca daily pagination metadata is invalid.")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("BLOCKED: Alpaca returned no formation daily bars.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True).dt.tz_convert(
        "America/New_York"
    )
    frame = frame.sort_values(["instrument_id", "timestamp"], kind="stable").reset_index(drop=True)
    local_dates = frame["timestamp"].dt.date
    if frame.assign(session_date=local_dates).duplicated(["instrument_id", "session_date"]).any():
        raise ValueError("Alpaca daily corpus contains duplicate stable-ID/session rows.")
    import exchange_calendars as xcals

    expected_dates = {
        value.date()
        for value in xcals.get_calendar("XNYS").sessions_in_range(formation_start, formation_end)
    }
    structural_errors = []
    for index, row in frame.iterrows():
        errors = validate_daily_observation(row, expected_dates=expected_dates)
        if errors and any(
            reason.startswith("daily timestamp") or "stable identity" in reason for reason in errors
        ):
            structural_errors.append({"row": int(index), "errors": errors})
    if structural_errors:
        raise ValueError(f"Alpaca daily corpus has structural errors: {structural_errors[:5]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    receipt = {
        "schema_version": "paper-v2-daily-acquisition-v1",
        "status": "complete",
        "request": request_identity,
        "pages": pages,
        "rows": len(frame),
        "symbols_requested": len(mapping),
        "symbols_observed": frame["symbol"].nunique(),
        "content_sha256": file_sha256(output_path),
        "rate_limit_headers": rate_limit,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(receipt_path, receipt)
    return receipt
