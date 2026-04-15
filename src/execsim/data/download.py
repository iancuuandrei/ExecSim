from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import os

import pandas as pd

from execsim.config import ExecSimConfig
from execsim.data.cleaning import clean_intraday_bars
from execsim.data.manifest import build_dataset_manifest
from execsim.data.validation import ValidationReport, validate_processed_bars


@dataclass(frozen=True, slots=True)
class SymbolPipelineResult:
    symbol: str
    raw_path: str
    processed_path: str
    raw_rows: int
    processed_rows: int
    validation_report: ValidationReport


@dataclass(frozen=True, slots=True)
class DataPipelineResult:
    symbols: tuple[SymbolPipelineResult, ...]
    manifest_path: str


def download_and_prepare_data(config: ExecSimConfig) -> DataPipelineResult:
    client = _create_alpaca_client()

    config.resolved_raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_processed_data_dir.mkdir(parents=True, exist_ok=True)

    results: list[SymbolPipelineResult] = []
    for symbol in config.symbols:
        raw_bars = download_raw_bars_for_symbol(client, config, symbol)
        raw_path = config.raw_symbol_path(symbol)
        raw_bars.to_parquet(raw_path, index=False)

        processed_bars = clean_intraday_bars(raw_bars, timezone=config.timezone)
        processed_path = config.processed_symbol_path(symbol)
        processed_bars.to_parquet(processed_path, index=False)

        report = validate_processed_bars(processed_bars, symbol=symbol)
        if not report.is_valid:
            raise ValueError("\n".join(report.to_lines()))

        results.append(
            SymbolPipelineResult(
                symbol=symbol,
                raw_path=str(raw_path),
                processed_path=str(processed_path),
                raw_rows=int(len(raw_bars)),
                processed_rows=int(len(processed_bars)),
                validation_report=report,
            )
        )

    build_dataset_manifest(config)
    return DataPipelineResult(
        symbols=tuple(results),
        manifest_path=str(config.resolved_manifest_path),
    )


def download_raw_bars_for_symbol(client: object, config: ExecSimConfig, symbol: str) -> pd.DataFrame:
    _ensure_supported_data_source(config)

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start_datetime, end_datetime = _resolve_datetime_range(config)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_datetime,
        end=end_datetime,
        adjustment=Adjustment(config.alpaca_adjustment.lower()),
        feed=DataFeed(config.alpaca_feed.lower()),
    )
    bars = client.get_stock_bars(request).df
    return _flatten_downloaded_bars(bars)


def _create_alpaca_client() -> object:
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment."
        )

    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError as exc:
        raise RuntimeError(
            "alpaca-py is required for download-data. Install project dependencies first."
        ) from exc

    return StockHistoricalDataClient(api_key, api_secret)


def _resolve_datetime_range(config: ExecSimConfig) -> tuple[datetime, datetime]:
    start_datetime = datetime.combine(config.start_date, time(0, 0), tzinfo=config.market_timezone)
    end_datetime = datetime.combine(
        config.end_date + timedelta(days=1),
        time(0, 0),
        tzinfo=config.market_timezone,
    )
    return start_datetime, end_datetime


def _flatten_downloaded_bars(bars: pd.DataFrame) -> pd.DataFrame:
    flattened = bars.copy()
    if "symbol" not in flattened.columns or "timestamp" not in flattened.columns:
        flattened = flattened.reset_index()

    if "symbol" not in flattened.columns:
        flattened["symbol"] = pd.Series(dtype="object")
    if "timestamp" not in flattened.columns:
        flattened["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")

    return flattened


def _ensure_supported_data_source(config: ExecSimConfig) -> None:
    if config.default_bar_timeframe != "1min":
        raise ValueError("download-data currently supports only 1-minute bars.")
    if config.data_provider.lower() != "alpaca":
        raise ValueError("download-data currently supports only data_provider=alpaca.")
