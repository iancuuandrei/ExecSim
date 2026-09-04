from __future__ import annotations

import pandas as pd

from execsim.config import ExecSimConfig
from execsim.data.loaders import load_processed_symbol_bars
from execsim.data.validation import summarize_daily_bar_counts

MANIFEST_COLUMNS = (
    "symbol",
    "first_timestamp",
    "last_timestamp",
    "n_rows",
    "n_days",
    "n_full_days_390",
    "min_date",
    "max_date",
)


def build_dataset_manifest(config: ExecSimConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in config.symbols:
        bars = load_processed_symbol_bars(config, symbol)
        rows.append(_build_manifest_row(symbol, bars))

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    output_path = config.resolved_manifest_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def _build_manifest_row(symbol: str, bars: pd.DataFrame) -> dict[str, object]:
    if bars.empty:
        return {
            "symbol": symbol,
            "first_timestamp": "",
            "last_timestamp": "",
            "n_rows": 0,
            "n_days": 0,
            "n_full_days_390": 0,
            "min_date": "",
            "max_date": "",
        }

    ordered = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    day_summaries = summarize_daily_bar_counts(ordered)
    session_dates = ordered["timestamp"].dt.date

    return {
        "symbol": symbol,
        "first_timestamp": ordered["timestamp"].iloc[0].isoformat(),
        "last_timestamp": ordered["timestamp"].iloc[-1].isoformat(),
        "n_rows": len(ordered),
        "n_days": len(day_summaries),
        "n_full_days_390": int(sum(item.is_full_day for item in day_summaries)),
        "min_date": session_dates.min().isoformat(),
        "max_date": session_dates.max().isoformat(),
    }
