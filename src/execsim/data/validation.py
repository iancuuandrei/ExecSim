from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from execsim.data.schema import (
    FULL_TRADING_DAY_BAR_COUNT,
    REQUIRED_COLUMNS,
    REQUIRED_NON_NULL_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class DailyBarCount:
    session_date: str
    bar_count: int

    @property
    def is_full_day(self) -> bool:
        return self.bar_count == FULL_TRADING_DAY_BAR_COUNT


@dataclass(frozen=True, slots=True)
class ValidationReport:
    symbol: str
    n_rows: int
    missing_columns: tuple[str, ...]
    duplicate_timestamps: int
    non_increasing_timestamps: int
    missing_required_values: dict[str, int]
    non_positive_volume_rows: int
    daily_bar_counts: tuple[DailyBarCount, ...]

    @property
    def empty_dataset(self) -> bool:
        return self.n_rows == 0

    @property
    def non_full_days(self) -> tuple[DailyBarCount, ...]:
        return tuple(item for item in self.daily_bar_counts if not item.is_full_day)

    @property
    def is_valid(self) -> bool:
        return (
            not self.empty_dataset
            and not self.missing_columns
            and self.duplicate_timestamps == 0
            and self.non_increasing_timestamps == 0
            and sum(self.missing_required_values.values()) == 0
            and self.non_positive_volume_rows == 0
        )

    def to_lines(self) -> list[str]:
        lines = [
            f"{self.symbol}: rows={self.n_rows}",
            f"  valid={self.is_valid}",
        ]

        if self.missing_columns:
            lines.append(f"  missing_columns={list(self.missing_columns)}")
        if self.empty_dataset:
            lines.append("  empty_dataset=True")
        if self.duplicate_timestamps:
            lines.append(f"  duplicate_timestamps={self.duplicate_timestamps}")
        if self.non_increasing_timestamps:
            lines.append(f"  non_increasing_timestamps={self.non_increasing_timestamps}")
        if any(self.missing_required_values.values()):
            lines.append(f"  missing_required_values={self.missing_required_values}")
        if self.non_positive_volume_rows:
            lines.append(f"  non_positive_volume_rows={self.non_positive_volume_rows}")
        if self.non_full_days:
            non_full_text = ", ".join(
                f"{item.session_date}:{item.bar_count}" for item in self.non_full_days
            )
            lines.append(f"  non_full_days={non_full_text}")

        return lines


def validate_processed_bars(bars: pd.DataFrame, symbol: str | None = None) -> ValidationReport:
    resolved_symbol = symbol or _resolve_symbol(bars)
    missing_columns = tuple(column for column in REQUIRED_COLUMNS if column not in bars.columns)

    if missing_columns:
        return ValidationReport(
            symbol=resolved_symbol,
            n_rows=len(bars),
            missing_columns=missing_columns,
            duplicate_timestamps=0,
            non_increasing_timestamps=0,
            missing_required_values={column: 0 for column in REQUIRED_NON_NULL_COLUMNS},
            non_positive_volume_rows=0,
            daily_bar_counts=(),
        )

    ordered = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicate_timestamps = int(ordered["timestamp"].duplicated().sum())
    timestamp_deltas = ordered["timestamp"].diff()
    non_increasing_timestamps = int(timestamp_deltas.iloc[1:].le(pd.Timedelta(0)).sum())

    missing_required_values = {
        column: int(ordered[column].isna().sum()) for column in REQUIRED_NON_NULL_COLUMNS
    }
    non_positive_volume_rows = int(ordered["volume"].le(0).sum())
    daily_bar_counts = summarize_daily_bar_counts(ordered)

    return ValidationReport(
        symbol=resolved_symbol,
        n_rows=len(ordered),
        missing_columns=missing_columns,
        duplicate_timestamps=duplicate_timestamps,
        non_increasing_timestamps=non_increasing_timestamps,
        missing_required_values=missing_required_values,
        non_positive_volume_rows=non_positive_volume_rows,
        daily_bar_counts=daily_bar_counts,
    )


def summarize_daily_bar_counts(bars: pd.DataFrame) -> tuple[DailyBarCount, ...]:
    if bars.empty:
        return ()

    daily_counts = (
        bars.assign(session_date=bars["timestamp"].dt.date)
        .groupby("session_date")
        .size()
        .reset_index(name="bar_count")
        .sort_values("session_date", kind="stable")
    )

    return tuple(
        DailyBarCount(session_date=row.session_date.isoformat(), bar_count=int(row.bar_count))
        for row in daily_counts.itertuples(index=False)
    )


def _resolve_symbol(bars: pd.DataFrame) -> str:
    if "symbol" not in bars.columns or bars.empty:
        return "<unknown>"

    unique_symbols = bars["symbol"].dropna().astype(str).str.upper().unique()
    if len(unique_symbols) == 1:
        return str(unique_symbols[0])

    return "MULTI"
