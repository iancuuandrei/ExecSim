from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    symbol: str
    side: str
    strategy: str
    requested_qty: int
    feasible_planned_qty: int
    filled_qty: int
    unfilled_qty: int
    average_fill_price: float | None
    arrival_price: float
    session_vwap: float | None
    implementation_shortfall: float | None
    implementation_shortfall_bps: float | None
    arrival_slippage_bps: float | None
    vwap_slippage_bps: float | None
    filled_notional: float
    completion_rate: float
    realized_participation: float
    average_participation: float
    maximum_participation: float
    modeled_spread_cost: float
    modeled_temporary_impact_cost: float
    total_modeled_execution_cost: float
    timing_cost: float | None
    incomplete_opportunity_cost: float | None
    cost_reconciliation_residual: float | None
    predicted_capacity_shortfall: int
    n_optimization_decisions: int
    optimizer_time_seconds: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    n_bars_in_window: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    summary: SimulationSummary
    execution_log: pd.DataFrame
    decision_trace: pd.DataFrame
