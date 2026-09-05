"""Validated schemas for the immutable paper corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

PAPER_BAR_COLUMNS = (
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
)


@dataclass(frozen=True, slots=True)
class PaperDataConfig:
    """Declare the locked corpus boundary and acquisition safeguards."""

    provider: Literal["alpaca"] = "alpaca"
    feed: Literal["sip"] = "sip"
    frequency: Literal["1min"] = "1min"
    timezone: Literal["America/New_York"] = "America/New_York"
    adjustment: Literal["raw"] = "raw"
    extended_hours: bool = False
    formation_start: date = date(2021, 1, 4)
    formation_end: date = date(2021, 12, 31)
    target_start: date = date(2022, 1, 3)
    target_end: date = date(2025, 12, 31)
    allow_network: bool = False
    paper_config_hash: str = ""

    def __post_init__(self) -> None:
        if self.extended_hours:
            raise ValueError("The primary paper corpus excludes extended-hours bars.")
        if not self.formation_start <= self.formation_end < self.target_start <= self.target_end:
            raise ValueError("Formation and target periods must be ordered and disjoint.")
        if self.paper_config_hash and len(self.paper_config_hash) != 64:
            raise ValueError("Paper config hash must be a full SHA-256 digest.")


@dataclass(frozen=True, slots=True)
class PaperUniverseMember:
    """Identify one frozen formation-period universe member."""

    rank: int
    instrument_id: str
    symbol: str
    median_price: float
    session_completeness: float
    median_daily_dollar_volume: float
    liquidity_group: int

    def __post_init__(self) -> None:
        if self.rank <= 0 or not self.instrument_id.strip() or not self.symbol.strip():
            raise ValueError("Universe members require rank, instrument_id, and symbol.")
        if self.median_price < 5.0:
            raise ValueError("Universe members must satisfy the formation-period $5 price floor.")
        if not 0.95 <= self.session_completeness <= 1.0:
            raise ValueError("Universe members must have at least 95% complete formation sessions.")
        if self.median_daily_dollar_volume <= 0 or not 1 <= self.liquidity_group <= 5:
            raise ValueError("Universe liquidity values are invalid.")


@dataclass(frozen=True, slots=True)
class AcquisitionChunk:
    """Describe one resumable monthly provider request."""

    instrument_id: str
    symbol: str
    start: date
    end: date
    feed: str = "sip"
    adjustment: str = "raw"

    @property
    def identity(self) -> str:
        return (
            f"{self.instrument_id}-{self.symbol.upper()}-{self.start:%Y%m%d}-"
            f"{self.end:%Y%m%d}-{self.feed}-{self.adjustment}"
        )


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    """Record a complete or failed immutable acquisition attempt."""

    chunk_identity: str
    status: Literal["complete", "failed"]
    requested_start: str
    requested_end: str
    feed: str
    adjustment: str
    downloaded_at: str
    response_sha256: str | None
    row_count: int
    instrument_id: str = ""
    symbol: str = ""
    observed_sessions: int = 0
    expected_sessions: int = 0
    error: str | None = None
    provider_request_id: str | None = None
    paper_config_hash: str = ""

    def __post_init__(self) -> None:
        if self.status == "complete" and (not self.response_sha256 or self.error):
            raise ValueError("Complete receipts require a hash and cannot contain an error.")
        if self.status == "failed" and (not self.error or self.response_sha256):
            raise ValueError("Failed receipts require an error and cannot claim a response hash.")
        if self.status == "complete" and (
            self.row_count <= 0
            or not self.instrument_id
            or not self.symbol
            or self.observed_sessions <= 0
            or self.expected_sessions <= 0
        ):
            raise ValueError("Complete receipts require validated identity and session coverage.")
        if self.paper_config_hash and len(self.paper_config_hash) != 64:
            raise ValueError("Acquisition receipt config hash must be SHA-256.")


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Carry immutable provider response bytes and auditable response metadata."""

    content: bytes
    row_count: int
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.content or self.row_count < 0:
            raise ValueError("Provider responses require non-empty bytes and non-negative rows.")


@dataclass(frozen=True, slots=True)
class InstrumentSymbolInterval:
    """Map one provider ticker interval to a stable instrument identity."""

    instrument_id: str
    symbol: str
    start: date
    end: date
    source: str

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.symbol or self.start > self.end or not self.source:
            raise ValueError("Ticker-history intervals require stable identity, dates, and source.")
