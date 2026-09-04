from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from execsim.ml.datasets.manifest import DatasetManifest

DatasetMode = Literal["static", "dynamic"]


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    mode: DatasetMode = "static"
    bucket_minutes: int = 5
    timezone: str = "America/New_York"
    exchange_calendar: str = "XNYS"
    require_calendar_complete: bool = True
    include_early_closes: bool = False
    rolling_sessions: int = 20
    data_classification: str = "historical"
    feature_schema_version: str = "volume-features-v1"
    target_schema_version: str = "volume-targets-v1"

    def __post_init__(self) -> None:
        if self.mode not in {"static", "dynamic"}:
            raise ValueError("Dataset mode must be static or dynamic.")
        if self.bucket_minutes not in {1, 5, 15}:
            raise ValueError("bucket_minutes must be 1, 5, or 15.")
        if self.rolling_sessions <= 0:
            raise ValueError("rolling_sessions must be positive.")


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    manifest: DatasetManifest
    manifest_path: Path
    rows: pd.DataFrame


def load_source_bars(paths: Sequence[str | Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to scan ML source datasets.") from exc
    resolved = [Path(path).resolve() for path in paths]
    source_hashes = {str(path): _file_hash(path) for path in resolved}
    dataset = ds.dataset([str(path) for path in resolved], format="parquet")
    columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    available = set(dataset.schema.names)
    table = dataset.scanner(
        columns=[column for column in columns if column in available]
    ).to_table()
    return table.to_pandas(), source_hashes


def iter_source_symbol_bars(
    paths: Sequence[str | Path],
) -> tuple[dict[str, str], Iterator[tuple[str, pd.DataFrame]]]:
    """Return source hashes and a lazy iterator of one-symbol pandas frames."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to scan ML source datasets.") from exc
    resolved = [Path(path).resolve() for path in paths]
    source_hashes = {str(path): _file_hash(path) for path in resolved}
    dataset = ds.dataset([str(path) for path in resolved], format="parquet")
    if "symbol" not in dataset.schema.names:
        raise ValueError("ML source Parquet must contain a symbol column.")
    symbols: set[str] = set()
    for batch in dataset.scanner(columns=["symbol"], batch_size=65_536).to_batches():
        symbols.update(str(value).upper() for value in batch.column(0).to_pylist())
    columns = [
        name
        for name in (
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
        if name in dataset.schema.names
    ]

    def frames() -> Iterator[tuple[str, pd.DataFrame]]:
        for symbol in sorted(symbols):
            table = dataset.scanner(
                columns=columns,
                filter=ds.field("symbol") == symbol,
                batch_size=65_536,
            ).to_table()
            yield symbol, table.to_pandas()

    return source_hashes, frames()


def build_dataset(
    *,
    output_root: str | Path,
    config: DatasetBuildConfig,
    bars: pd.DataFrame | None = None,
    source_paths: Sequence[str | Path] = (),
    materialize_result_rows: bool = True,
) -> DatasetBuildResult:
    if bars is None:
        if not source_paths:
            raise ValueError("Provide bars or at least one source Parquet path.")
        source_hashes, source_frames = iter_source_symbol_bars(source_paths)
    else:
        source_hashes = {"in_memory": _frame_hash(bars)}
        source_frames = iter((("in_memory", bars),))
    config_payload = {
        "mode": config.mode,
        "bucket_minutes": config.bucket_minutes,
        "timezone": config.timezone,
        "exchange_calendar": config.exchange_calendar,
        "require_calendar_complete": config.require_calendar_complete,
        "include_early_closes": config.include_early_closes,
        "rolling_sessions": config.rolling_sessions,
        "data_classification": config.data_classification,
        "feature_schema_version": config.feature_schema_version,
        "target_schema_version": config.target_schema_version,
        "source_hashes": source_hashes,
    }
    dataset_id = (
        "volume-"
        + hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:12]
    )
    dataset_dir = Path(output_root) / dataset_id
    partitions: list[str] = []
    exclusions: list[dict[str, object]] = []
    materialized: list[pd.DataFrame] = []
    all_dates: set[str] = set()
    all_samples: set[str] = set()
    all_symbols: set[str] = set()
    row_count = 0
    for _, source_frame in source_frames:
        prepared = _prepare_bars(source_frame, config.timezone)
        bucketed = _bucket_bars(prepared, config.bucket_minutes)
        frame_rows, frame_exclusions = _build_rows(bucketed, config)
        exclusions.extend(frame_exclusions)
        if frame_rows.empty:
            continue
        for symbol, group in frame_rows.groupby("symbol", sort=True):
            path = dataset_dir / f"mode={config.mode}" / f"symbol={symbol}" / "part-000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            group = group.reset_index(drop=True)
            group.to_parquet(path, index=False)
            partitions.append(str(path.relative_to(dataset_dir).as_posix()))
            row_count += len(group)
            all_symbols.add(str(symbol))
            all_dates.update(group["session_date"].astype(str))
            all_samples.update(group["sample_id"].astype(str))
            if materialize_result_rows:
                materialized.append(group)
    if not row_count:
        raise ValueError("Dataset build produced no point-in-time samples.")
    rows = pd.concat(materialized, ignore_index=True) if materialized else pd.DataFrame()
    dates = sorted(all_dates)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        mode=config.mode,
        bucket_minutes=config.bucket_minutes,
        timezone=config.timezone,
        feature_schema_version=config.feature_schema_version,
        target_schema_version=config.target_schema_version,
        data_classification=config.data_classification,
        row_count=row_count,
        sample_count=len(all_samples),
        symbol_count=len(all_symbols),
        symbols=tuple(sorted(all_symbols)),
        min_session_date=str(dates[0]),
        max_session_date=str(dates[-1]),
        source_hashes=source_hashes,
        partitions=tuple(partitions),
        filters={
            "calendar_complete": config.require_calendar_complete,
            "include_early_closes": config.include_early_closes,
        },
        exclusions=tuple(exclusions),
        git_commit=_git_commit(),
        built_at=datetime.now(UTC).isoformat(),
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest.write(manifest_path)
    return DatasetBuildResult(manifest, manifest_path, rows)


def _prepare_bars(bars: pd.DataFrame, timezone: str) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"ML source bars missing columns: {sorted(missing)}")
    prepared = bars.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    if prepared["timestamp"].dt.tz is None or str(prepared["timestamp"].dt.tz) != timezone:
        raise ValueError(f"ML source timestamps must use timezone {timezone}.")
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
    prepared = prepared.sort_values(["symbol", "timestamp"], kind="stable")
    if prepared.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("ML source bars contain duplicate symbol timestamps.")
    return prepared.reset_index(drop=True)


def _bucket_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    prepared = bars.copy()
    prepared["bucket_timestamp"] = prepared["timestamp"].dt.floor(f"{minutes}min")
    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "trade_count" in prepared:
        aggregation["trade_count"] = "sum"
    grouped = (
        prepared.groupby(["symbol", "bucket_timestamp"], sort=True)
        .agg(aggregation)
        .reset_index()
        .rename(columns={"bucket_timestamp": "timestamp"})
    )
    return grouped


def _build_rows(
    bars: pd.DataFrame, config: DatasetBuildConfig
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for symbol, symbol_bars in bars.groupby("symbol", sort=True):
        sessions = [
            group.reset_index(drop=True)
            for _, group in symbol_bars.groupby(symbol_bars["timestamp"].dt.date, sort=True)
        ]
        valid: list[pd.DataFrame] = []
        for session in sessions:
            reason = _session_exclusion_reason(session, config)
            if reason:
                exclusions.append(
                    {
                        "symbol": symbol,
                        "session_date": str(session["timestamp"].dt.date.iloc[0]),
                        "reason": reason,
                    }
                )
            else:
                valid.append(session)
        for index in range(1, len(valid)):
            current = valid[index]
            history = valid[max(0, index - config.rolling_sessions) : index]
            rows.extend(
                _static_rows(symbol, current, history)
                if config.mode == "static"
                else _dynamic_rows(symbol, current, history)
            )
    return pd.DataFrame(rows), exclusions


def _session_exclusion_reason(session: pd.DataFrame, config: DatasetBuildConfig) -> str | None:
    if not config.require_calendar_complete:
        return None
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar(config.exchange_calendar)
        session_date = pd.Timestamp(session["timestamp"].dt.date.iloc[0])
        if not calendar.is_session(session_date):
            return "not_exchange_session"
        opened = calendar.session_open(session_date)
        closed = calendar.session_close(session_date)
        duration_minutes = int((closed - opened).total_seconds() / 60)
        is_early_close = duration_minutes < 390
        if is_early_close and not config.include_early_closes:
            return "early_close_excluded"
        expected = int(np.ceil(duration_minutes / config.bucket_minutes))
        if len(session) != expected:
            return f"incomplete_session:{len(session)}_of_{expected}_buckets"
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'data' extra for calendar-aware ML datasets.") from exc
    return None


def _history_features(history: list[pd.DataFrame]) -> dict[str, float]:
    totals = np.array([float(frame["volume"].sum()) for frame in history])
    previous = history[-1]
    closes = np.array([float(frame["close"].iloc[-1]) for frame in history])
    close_returns = np.diff(np.log(closes))
    trade_count = float(previous["trade_count"].sum()) if "trade_count" in previous else 0.0
    return {
        "previous_session_total_volume": float(totals[-1]),
        "rolling_adv": float(totals.mean()),
        "rolling_median_volume": float(np.median(totals)),
        "rolling_volatility": float(close_returns.std()) if len(close_returns) else 0.0,
        "previous_session_return": float(
            np.log(previous["close"].iloc[-1] / previous["open"].iloc[0])
        ),
        "previous_session_range": float(
            (previous["high"].max() - previous["low"].min()) / previous["open"].iloc[0]
        ),
        "previous_session_trade_count": trade_count,
    }


def _static_rows(
    symbol: str, current: pd.DataFrame, history: list[pd.DataFrame]
) -> list[dict[str, object]]:
    session_date = current["timestamp"].dt.date.iloc[0]
    as_of = current["timestamp"].iloc[0]
    total = float(current["volume"].sum())
    features = _history_features(history)
    output = []
    for bucket_index, row in enumerate(current.itertuples(index=False)):
        output.append(
            {
                "sample_id": f"{symbol}|{session_date}|static",
                "mode": "static",
                "symbol": symbol,
                "session_date": session_date.isoformat(),
                "as_of": as_of,
                "target_bucket_timestamp": row.timestamp,
                "target_bucket_index": bucket_index,
                "weekday": float(session_date.weekday()),
                "month": float(session_date.month),
                "bucket_index": float(bucket_index),
                **features,
                "target_log_total_volume": float(np.log1p(total)),
                "target_volume_share": float(row.volume / total) if total else 0.0,
                "target_remaining_volume": total,
                "target_conditional_share": float(row.volume / total) if total else 0.0,
                "feature_available_at": history[-1]["timestamp"].iloc[-1],
            }
        )
    return output


def _dynamic_rows(
    symbol: str, current: pd.DataFrame, history: list[pd.DataFrame]
) -> list[dict[str, object]]:
    session_date = current["timestamp"].dt.date.iloc[0]
    base = _history_features(history)
    total_buckets = len(current)
    output = []
    for decision_index in range(1, total_buckets):
        observed = current.iloc[:decision_index]
        future = current.iloc[decision_index:]
        remaining_total = float(future["volume"].sum())
        recent_returns = np.log(observed["close"]).diff().dropna().tail(5)
        dynamic = {
            "elapsed_session_fraction": decision_index / total_buckets,
            "observed_cumulative_volume": float(observed["volume"].sum()),
            "recent_bucket_volume": float(observed["volume"].iloc[-1]),
            "recent_realized_volatility": float(recent_returns.std(ddof=0))
            if len(recent_returns)
            else 0.0,
        }
        as_of = current["timestamp"].iloc[decision_index]
        sample_id = f"{symbol}|{session_date}|dynamic|{decision_index:04d}"
        for target_index, row in zip(
            range(decision_index, total_buckets), future.itertuples(index=False), strict=True
        ):
            output.append(
                {
                    "sample_id": sample_id,
                    "mode": "dynamic",
                    "symbol": symbol,
                    "session_date": session_date.isoformat(),
                    "as_of": as_of,
                    "target_bucket_timestamp": row.timestamp,
                    "target_bucket_index": target_index,
                    "weekday": float(session_date.weekday()),
                    "month": float(session_date.month),
                    "bucket_index": float(target_index),
                    **base,
                    **dynamic,
                    "target_log_total_volume": float(np.log1p(current["volume"].sum())),
                    "target_volume_share": float(row.volume / current["volume"].sum()),
                    "target_remaining_volume": remaining_total,
                    "target_conditional_share": float(row.volume / remaining_total)
                    if remaining_total
                    else 0.0,
                    "feature_available_at": observed["timestamp"].iloc[-1],
                }
            )
    return output


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1).to_csv(index=False)
    return hashlib.sha256(ordered.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None
