from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ChildFill:
    symbol: str
    timestamp: pd.Timestamp
    side: str
    scheduled_qty: int
    max_allowed_qty: int
    filled_qty: int
    bar_volume: float
    fill_price: float


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    symbol: str
    side: str
    requested_qty: int
    filled_qty: int
    unfilled_qty: int
    average_fill_price: float | None
    arrival_price: float
    session_vwap: float | None
    implementation_shortfall_bps: float | None
    vwap_slippage_bps: float | None
    filled_notional: float
    completion_rate: float
    realized_participation: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    n_bars_in_window: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    summary: SimulationSummary
    execution_log: pd.DataFrame
