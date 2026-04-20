from __future__ import annotations

import math

import pandas as pd

from execsim.orders import ParentOrder
from execsim.simulator.models import SimulationResult, SimulationSummary
from execsim.strategies.base import SchedulingStrategy
from execsim.strategies.twap import TwapStrategy

REQUIRED_SIMULATION_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
EXECUTION_LOG_COLUMNS = (
    "symbol",
    "timestamp",
    "side",
    "scheduled_qty",
    "max_allowed_qty",
    "filled_qty",
    "bar_volume",
    "fill_price",
)


def simulate_twap(
    parent_order: ParentOrder,
    bars: pd.DataFrame,
    max_bar_participation_rate: float,
) -> SimulationResult:
    return simulate_order(
        parent_order=parent_order,
        bars=bars,
        strategy=TwapStrategy(),
        max_bar_participation_rate=max_bar_participation_rate,
    )


def simulate_order(
    parent_order: ParentOrder,
    bars: pd.DataFrame,
    strategy: SchedulingStrategy,
    max_bar_participation_rate: float,
) -> SimulationResult:
    _validate_participation_rate(max_bar_participation_rate)
    session_bars = _prepare_session_bars(parent_order, bars)
    window_bars = _prepare_window_bars(parent_order, bars)
    schedule = strategy.generate_schedule(parent_order, window_bars)
    scheduled_quantities = _extract_scheduled_quantities(schedule, len(window_bars))

    records: list[dict[str, object]] = []
    remaining_qty = parent_order.quantity
    weighted_fill_notional = 0.0
    filled_qty = 0

    for row_index, row in window_bars.iterrows():
        scheduled_qty = scheduled_quantities[row_index]
        bar_volume = float(row["volume"])
        max_allowed_qty = max(0, math.floor(max_bar_participation_rate * bar_volume))
        child_fill_qty = min(scheduled_qty, remaining_qty, max_allowed_qty)
        fill_price = _resolve_fill_price(row)

        records.append(
            {
                "symbol": parent_order.symbol,
                "timestamp": row["timestamp"],
                "side": parent_order.side,
                "scheduled_qty": scheduled_qty,
                "max_allowed_qty": max_allowed_qty,
                "filled_qty": child_fill_qty,
                "bar_volume": bar_volume,
                "fill_price": fill_price,
            }
        )

        if child_fill_qty > 0:
            filled_qty += child_fill_qty
            remaining_qty -= child_fill_qty
            weighted_fill_notional += child_fill_qty * fill_price

    execution_log = pd.DataFrame.from_records(records, columns=EXECUTION_LOG_COLUMNS)
    total_window_volume = float(execution_log["bar_volume"].sum())
    average_fill_price = (
        weighted_fill_notional / filled_qty if filled_qty > 0 else None
    )
    arrival_price = _arrival_price(window_bars)
    session_vwap = _session_vwap(session_bars)
    side_multiplier = _side_multiplier(parent_order.side)
    implementation_shortfall_bps = _signed_slippage_bps(
        average_fill_price,
        arrival_price,
        side_multiplier,
    )
    vwap_slippage_bps = _signed_slippage_bps(
        average_fill_price,
        session_vwap,
        side_multiplier,
    )

    summary = SimulationSummary(
        symbol=parent_order.symbol,
        side=parent_order.side,
        requested_qty=parent_order.quantity,
        filled_qty=filled_qty,
        unfilled_qty=parent_order.quantity - filled_qty,
        average_fill_price=average_fill_price,
        arrival_price=arrival_price,
        session_vwap=session_vwap,
        implementation_shortfall_bps=implementation_shortfall_bps,
        vwap_slippage_bps=vwap_slippage_bps,
        filled_notional=weighted_fill_notional,
        completion_rate=filled_qty / parent_order.quantity,
        realized_participation=(
            filled_qty / total_window_volume if total_window_volume > 0 else 0.0
        ),
        start_timestamp=window_bars["timestamp"].iloc[0],
        end_timestamp=window_bars["timestamp"].iloc[-1],
        n_bars_in_window=len(window_bars),
    )

    return SimulationResult(summary=summary, execution_log=execution_log)


def _validate_participation_rate(value: float) -> None:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError("max_bar_participation_rate must be numeric.")
    if value < 0 or value > 1:
        raise ValueError("max_bar_participation_rate must be between 0 and 1.")


def _prepare_session_bars(parent_order: ParentOrder, bars: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_SIMULATION_COLUMNS if column not in bars.columns
    ]
    if missing_columns:
        raise ValueError(f"Simulation bars missing required columns: {missing_columns}")

    prepared = bars.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])

    if "symbol" in prepared.columns:
        symbol_mask = prepared["symbol"].astype(str).str.upper() == parent_order.symbol
        prepared = prepared.loc[symbol_mask].copy()

    prepared = (
        prepared.loc[prepared["timestamp"].dt.date == parent_order.trade_date]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    if prepared.empty:
        raise ValueError(
            "No processed bars found for "
            f"{parent_order.symbol} on {parent_order.trade_date.isoformat()}."
        )

    return prepared


def _prepare_window_bars(parent_order: ParentOrder, bars: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_SIMULATION_COLUMNS if column not in bars.columns
    ]
    if missing_columns:
        raise ValueError(f"Simulation bars missing required columns: {missing_columns}")

    prepared = bars.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])

    if "symbol" in prepared.columns:
        symbol_mask = prepared["symbol"].astype(str).str.upper() == parent_order.symbol
        prepared = prepared.loc[symbol_mask].copy()

    timestamps = prepared["timestamp"]
    times = timestamps.dt.time
    window_mask = (
        (timestamps.dt.date == parent_order.trade_date)
        & (times >= parent_order.start_time)
        & (times < parent_order.end_time)
    )
    prepared = (
        prepared.loc[window_mask]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    if prepared.empty:
        raise ValueError(
            "No processed bars found for "
            f"{parent_order.symbol} on {parent_order.trade_date.isoformat()} "
            f"between {parent_order.start_time} and {parent_order.end_time}."
        )

    return prepared


def _extract_scheduled_quantities(schedule: pd.DataFrame, expected_length: int) -> list[int]:
    if "scheduled_qty" not in schedule.columns:
        raise ValueError("Strategy schedule must include scheduled_qty.")
    if len(schedule) != expected_length:
        raise ValueError("Strategy schedule length must match the bar window length.")

    quantities = [int(quantity) for quantity in schedule["scheduled_qty"].tolist()]
    if any(quantity < 0 for quantity in quantities):
        raise ValueError("Strategy schedule quantities must be non-negative.")

    return quantities


def _resolve_fill_price(row: pd.Series) -> float:
    return _bar_price(row)


def _arrival_price(window_bars: pd.DataFrame) -> float:
    return _bar_price(window_bars.iloc[0])


def _session_vwap(session_bars: pd.DataFrame) -> float | None:
    total_volume = float(session_bars["volume"].sum())
    if total_volume <= 0:
        return None

    weighted_notional = sum(
        float(row["volume"]) * _bar_price(row)
        for _, row in session_bars.iterrows()
    )
    return weighted_notional / total_volume


def _signed_slippage_bps(
    execution_price: float | None,
    benchmark_price: float | None,
    side_multiplier: int,
) -> float | None:
    if execution_price is None or benchmark_price is None or benchmark_price == 0:
        return None
    return 10_000 * side_multiplier * (execution_price - benchmark_price) / benchmark_price


def _side_multiplier(side: str) -> int:
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    raise ValueError("Side must be 'buy' or 'sell'.")


def _bar_price(row: pd.Series) -> float:
    vwap = row.get("vwap")
    if vwap is not None and not pd.isna(vwap):
        return float(vwap)

    return (
        float(row["open"])
        + float(row["high"])
        + float(row["low"])
        + float(row["close"])
    ) / 4.0
