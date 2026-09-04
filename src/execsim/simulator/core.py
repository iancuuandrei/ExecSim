from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from execsim.costs import ExecutionCostModel, LinearTemporaryImpactModel
from execsim.forecasting.models import VolumeForecast, VolumeForecastProvider
from execsim.orders import ParentOrder
from execsim.policies import DecisionContext, ExecutionConstraints, PovPolicy, TwapPolicy
from execsim.policies.models import PolicyDecision, SchedulePlan
from execsim.simulator.models import SimulationResult, SimulationSummary
from execsim.strategies.base import SchedulingStrategy

REQUIRED_SIMULATION_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
EXECUTION_LOG_COLUMNS = (
    "symbol",
    "session_date",
    "timestamp",
    "side",
    "strategy",
    "requested_parent_qty",
    "inventory_before",
    "planned_qty",
    "scheduled_qty",
    "actual_market_volume",
    "forecast_market_volume",
    "volume_forecast_error",
    "volume_vs_forecast",
    "bar_volume",
    "planned_participation",
    "actual_capacity",
    "max_allowed_qty",
    "executed_qty",
    "filled_qty",
    "inventory_after",
    "reference_price",
    "half_spread",
    "temporary_impact_per_share",
    "execution_price",
    "fill_price",
    "spread_cost",
    "temporary_impact_cost",
    "timing_cost",
    "cumulative_executed_qty",
    "cumulative_modeled_cost",
    "remaining_parent_qty",
    "forecast_id",
    "decision_id",
)


@dataclass(frozen=True, slots=True)
class _LegacySchedulePolicy:
    strategy: SchedulingStrategy
    policy_name: str = "custom"

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        bars = pd.DataFrame({"timestamp": context.future_timestamps})
        schedule = self.strategy.generate_schedule(parent_order, bars)
        quantities = _extract_scheduled_quantities(schedule, len(bars))
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=context.future_timestamps,
            quantities=tuple(quantities),
            feasible_planned_quantity=sum(quantities),
        )


def simulate_twap(
    parent_order: ParentOrder,
    bars: pd.DataFrame,
    max_bar_participation_rate: float,
    cost_model: ExecutionCostModel | None = None,
) -> SimulationResult:
    return simulate_policy(
        parent_order=parent_order,
        bars=bars,
        policy=TwapPolicy(),
        constraints=ExecutionConstraints(
            planned_participation_rate=max_bar_participation_rate,
            hard_participation_rate=max_bar_participation_rate,
        ),
        cost_model=cost_model,
    )


def simulate_order(
    parent_order: ParentOrder,
    bars: pd.DataFrame,
    strategy: SchedulingStrategy,
    max_bar_participation_rate: float,
    cost_model: ExecutionCostModel | None = None,
) -> SimulationResult:
    """Compatibility adapter for the original schedule strategy protocol."""
    return simulate_policy(
        parent_order=parent_order,
        bars=bars,
        policy=_LegacySchedulePolicy(strategy),
        constraints=ExecutionConstraints(
            planned_participation_rate=max_bar_participation_rate,
            hard_participation_rate=max_bar_participation_rate,
        ),
        cost_model=cost_model,
    )


def simulate_policy(
    *,
    parent_order: ParentOrder,
    bars: pd.DataFrame,
    policy: Any,
    constraints: ExecutionConstraints | None = None,
    cost_model: ExecutionCostModel | None = None,
    forecast_provider: VolumeForecastProvider | None = None,
) -> SimulationResult:
    constraints = constraints or ExecutionConstraints()
    active_cost_model: ExecutionCostModel = cost_model or LinearTemporaryImpactModel()
    session_bars = _prepare_session_bars(parent_order, bars, constraints.timezone)
    window_bars = _prepare_window_bars(parent_order, session_bars)
    timestamps = tuple(pd.Timestamp(value) for value in window_bars["timestamp"])
    adaptive = hasattr(policy, "decide")
    if hasattr(policy, "reset"):
        policy.reset()

    static_plan: SchedulePlan | None = None
    static_forecast: VolumeForecast | None = None
    if not adaptive:
        static_forecast = _make_forecast(
            forecast_provider, parent_order, timestamps[0], timestamps, None
        )
        context = _decision_context(
            timestamp=timestamps[0],
            remaining=parent_order.quantity,
            elapsed=0,
            timestamps=timestamps,
            observations=window_bars.iloc[:0].copy(),
            forecast=static_forecast,
            constraints=constraints,
        )
        static_plan = policy.create_plan(parent_order, context)

    reference_prices = _bar_prices(window_bars)
    arrival_price = float(reference_prices[0])
    records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    remaining = parent_order.quantity
    cumulative_executed = 0
    cumulative_modeled_cost = 0.0
    predicted_shortfall = static_plan.predicted_capacity_shortfall if static_plan else 0

    for index, timestamp in enumerate(timestamps):
        row = window_bars.iloc[index]
        volume = float(row["volume"])
        forecast_id = static_plan.forecast_id if static_plan else None
        forecast_market_volume: float | None = None
        decision_id: str | None = None
        if static_plan is not None:
            planned = int(static_plan.quantities[index])
            if static_forecast is not None:
                forecast_market_volume = static_forecast.expected_volumes[index]
        elif remaining == 0:
            planned = 0
        else:
            future = timestamps[index:]
            observations = window_bars.iloc[:index].copy()
            forecast = _make_forecast(
                forecast_provider, parent_order, timestamp, future, observations
            )
            if forecast is not None:
                forecast_market_volume = forecast.expected_volumes[0]
            context = _decision_context(
                timestamp=timestamp,
                remaining=remaining,
                elapsed=index,
                timestamps=future,
                observations=observations,
                forecast=forecast,
                constraints=constraints,
            )
            decision = (
                policy.decide(context, observable_current_volume=volume)
                if isinstance(policy, PovPolicy)
                else policy.decide(context)
            )
            if not isinstance(decision, PolicyDecision):
                raise TypeError("Adaptive policies must return PolicyDecision.")
            planned = decision.planned_quantity
            forecast_id = decision.forecast_id
            decision_id = decision.decision_id
            if decision.trace is not None:
                traces.append(decision.trace)
                predicted_shortfall = max(
                    predicted_shortfall,
                    _trace_integer(decision.trace, "predicted_capacity_shortfall"),
                )

        actual_capacity = max(0, math.floor(constraints.hard_participation_rate * volume))
        executed = min(planned, remaining, actual_capacity)
        quote = active_cost_model.quote(
            side=parent_order.side,
            reference_price=float(reference_prices[index]),
            executed_qty=int(executed),
            market_volume=volume,
        )
        inventory_before = remaining
        remaining -= executed
        cumulative_executed += executed
        cumulative_modeled_cost += quote.total_modeled_cost
        timing_cost = (
            _side_multiplier(parent_order.side) * executed * (quote.reference_price - arrival_price)
        )
        records.append(
            {
                "symbol": parent_order.symbol,
                "session_date": parent_order.trade_date.isoformat(),
                "timestamp": timestamp,
                "side": parent_order.side,
                "strategy": getattr(policy, "policy_name", type(policy).__name__),
                "requested_parent_qty": parent_order.quantity,
                "inventory_before": inventory_before,
                "planned_qty": planned,
                "scheduled_qty": planned,
                "actual_market_volume": volume,
                "forecast_market_volume": forecast_market_volume,
                "volume_forecast_error": (
                    volume - forecast_market_volume if forecast_market_volume is not None else None
                ),
                "volume_vs_forecast": (
                    "above"
                    if forecast_market_volume is not None and volume > forecast_market_volume
                    else "below"
                    if forecast_market_volume is not None and volume < forecast_market_volume
                    else "equal"
                    if forecast_market_volume is not None
                    else "not_available"
                ),
                "bar_volume": volume,
                "planned_participation": planned / volume if volume > 0 else 0.0,
                "actual_capacity": actual_capacity,
                "max_allowed_qty": actual_capacity,
                "executed_qty": executed,
                "filled_qty": executed,
                "inventory_after": remaining,
                "reference_price": quote.reference_price,
                "half_spread": quote.half_spread,
                "temporary_impact_per_share": quote.temporary_impact_per_share,
                "execution_price": quote.execution_price,
                "fill_price": quote.execution_price,
                "spread_cost": quote.spread_cost,
                "temporary_impact_cost": quote.temporary_impact_cost,
                "timing_cost": timing_cost,
                "cumulative_executed_qty": cumulative_executed,
                "cumulative_modeled_cost": cumulative_modeled_cost,
                "remaining_parent_qty": remaining,
                "forecast_id": forecast_id,
                "decision_id": decision_id,
            }
        )

    execution_log = pd.DataFrame.from_records(records, columns=EXECUTION_LOG_COLUMNS)
    decision_trace = pd.DataFrame.from_records(traces)
    summary = _summarize(
        parent_order=parent_order,
        session_bars=session_bars,
        window_bars=window_bars,
        execution_log=execution_log,
        strategy=str(execution_log["strategy"].iloc[0]),
        feasible_planned=(
            static_plan.feasible_planned_quantity if static_plan else parent_order.quantity
        ),
        predicted_shortfall=predicted_shortfall,
        static_plan=static_plan,
        decision_trace=decision_trace,
    )
    return SimulationResult(
        summary=summary, execution_log=execution_log, decision_trace=decision_trace
    )


def _decision_context(
    *,
    timestamp: pd.Timestamp,
    remaining: int,
    elapsed: int,
    timestamps: tuple[pd.Timestamp, ...],
    observations: pd.DataFrame,
    forecast: VolumeForecast | None,
    constraints: ExecutionConstraints,
) -> DecisionContext:
    return DecisionContext(
        current_timestamp=timestamp,
        decision_timing="bucket_start",
        remaining_inventory=remaining,
        elapsed_buckets=elapsed,
        remaining_buckets=len(timestamps),
        observations=observations,
        future_timestamps=timestamps,
        forecast=forecast,
        constraints=constraints,
    )


def _make_forecast(
    provider: VolumeForecastProvider | None,
    order: ParentOrder,
    generated_at: pd.Timestamp,
    timestamps: tuple[pd.Timestamp, ...],
    observations: pd.DataFrame | None,
) -> VolumeForecast | None:
    if provider is None:
        return None
    return provider.forecast(
        symbol=order.symbol,
        session_date=order.trade_date,
        generated_at=generated_at,
        bucket_timestamps=timestamps,
        observations=observations,
    )


def _prepare_session_bars(
    parent_order: ParentOrder, bars: pd.DataFrame, timezone: str
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SIMULATION_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"Simulation bars missing required columns: {missing}")
    prepared = bars.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    if prepared["timestamp"].dt.tz is None:
        raise ValueError("Simulation timestamps must be timezone-aware.")
    if str(prepared["timestamp"].dt.tz) != timezone:
        raise ValueError(f"Simulation timestamps must use configured timezone {timezone}.")
    if "symbol" in prepared.columns:
        prepared = prepared.loc[
            prepared["symbol"].astype(str).str.upper() == parent_order.symbol
        ].copy()
    prepared = prepared.loc[prepared["timestamp"].dt.date == parent_order.trade_date].copy()
    if prepared.empty:
        raise ValueError(
            f"No processed bars found for {parent_order.symbol} on {parent_order.trade_date}."
        )
    prepared = prepared.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if prepared["timestamp"].duplicated().any():
        raise ValueError("Simulation bars contain duplicate timestamps.")
    columns = ["open", "high", "low", "close", "volume"]
    numeric = prepared.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Simulation OHLCV values must be finite.")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Simulation OHLC prices must be positive.")
    if (numeric["volume"] < 0).any():
        raise ValueError("Simulation volume must be non-negative.")
    prepared[columns] = numeric
    return prepared


def _prepare_window_bars(parent_order: ParentOrder, session_bars: pd.DataFrame) -> pd.DataFrame:
    times = session_bars["timestamp"].dt.time
    window = session_bars.loc[
        (times >= parent_order.start_time) & (times < parent_order.end_time)
    ].reset_index(drop=True)
    if window.empty:
        raise ValueError(
            "No processed bars found between "
            f"{parent_order.start_time} and {parent_order.end_time}."
        )
    return window


def _bar_prices(bars: pd.DataFrame) -> np.ndarray:
    ohlc = bars.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    fallback = ohlc.mean(axis=1)
    if "vwap" not in bars.columns:
        return fallback
    vwap = pd.to_numeric(bars["vwap"], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(vwap) & (vwap > 0), vwap, fallback)


def _summarize(
    *,
    parent_order: ParentOrder,
    session_bars: pd.DataFrame,
    window_bars: pd.DataFrame,
    execution_log: pd.DataFrame,
    strategy: str,
    feasible_planned: int,
    predicted_shortfall: int,
    static_plan: SchedulePlan | None,
    decision_trace: pd.DataFrame,
) -> SimulationSummary:
    quantities = execution_log["executed_qty"].to_numpy(dtype=float)
    prices = execution_log["execution_price"].to_numpy(dtype=float)
    references = execution_log["reference_price"].to_numpy(dtype=float)
    volumes = execution_log["actual_market_volume"].to_numpy(dtype=float)
    filled = int(quantities.sum())
    filled_notional = float(np.dot(quantities, prices))
    average = filled_notional / filled if filled else None
    arrival = float(references[0])
    side = _side_multiplier(parent_order.side)
    session_prices = _bar_prices(session_bars)
    session_volumes = session_bars["volume"].to_numpy(dtype=float)
    session_total = float(session_volumes.sum())
    session_vwap = (
        float(np.dot(session_prices, session_volumes) / session_total) if session_total else None
    )
    implementation_shortfall = (
        float(side * np.dot(quantities, prices - arrival)) if filled else None
    )
    timing_cost = float(side * np.dot(quantities, references - arrival)) if filled else None
    spread_cost = float(execution_log["spread_cost"].sum())
    impact_cost = float(execution_log["temporary_impact_cost"].sum())
    residual = (
        implementation_shortfall - timing_cost - spread_cost - impact_cost
        if implementation_shortfall is not None and timing_cost is not None
        else None
    )
    unfilled = parent_order.quantity - filled
    incomplete = side * unfilled * (float(references[-1]) - arrival) if unfilled else 0.0
    participation = np.divide(quantities, volumes, out=np.zeros_like(quantities), where=volumes > 0)
    total_window_volume = float(volumes.sum())
    solver_time = 0.0
    n_optimizations = 0
    if static_plan is not None and static_plan.solver_diagnostics is not None:
        solver_time = float(getattr(static_plan.solver_diagnostics, "solve_time_seconds", 0.0))
        n_optimizations = 1
    elif not decision_trace.empty and "solve_time_seconds" in decision_trace:
        solver_time = float(decision_trace["solve_time_seconds"].sum())
        n_optimizations = len(decision_trace)
    shortfall_bps = _signed_slippage_bps(average, arrival, side)
    return SimulationSummary(
        symbol=parent_order.symbol,
        side=parent_order.side,
        strategy=strategy,
        requested_qty=parent_order.quantity,
        feasible_planned_qty=feasible_planned,
        filled_qty=filled,
        unfilled_qty=unfilled,
        average_fill_price=average,
        arrival_price=arrival,
        session_vwap=session_vwap,
        implementation_shortfall=implementation_shortfall,
        implementation_shortfall_bps=shortfall_bps,
        arrival_slippage_bps=shortfall_bps,
        vwap_slippage_bps=_signed_slippage_bps(average, session_vwap, side),
        filled_notional=filled_notional,
        completion_rate=filled / parent_order.quantity,
        realized_participation=filled / total_window_volume if total_window_volume else 0.0,
        average_participation=float(participation.mean()) if len(participation) else 0.0,
        maximum_participation=float(participation.max(initial=0.0)),
        modeled_spread_cost=spread_cost,
        modeled_temporary_impact_cost=impact_cost,
        total_modeled_execution_cost=spread_cost + impact_cost,
        timing_cost=timing_cost,
        incomplete_opportunity_cost=float(incomplete),
        cost_reconciliation_residual=residual,
        predicted_capacity_shortfall=predicted_shortfall,
        n_optimization_decisions=n_optimizations,
        optimizer_time_seconds=solver_time,
        start_timestamp=pd.Timestamp(window_bars["timestamp"].iloc[0]),
        end_timestamp=pd.Timestamp(window_bars["timestamp"].iloc[-1]),
        n_bars_in_window=len(window_bars),
    )


def _extract_scheduled_quantities(schedule: pd.DataFrame, expected_length: int) -> list[int]:
    if "scheduled_qty" not in schedule.columns or len(schedule) != expected_length:
        raise ValueError("Strategy schedule must match bars and include scheduled_qty.")
    raw = schedule["scheduled_qty"].tolist()
    if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in raw):
        raise ValueError("Strategy schedule quantities must be non-negative integers.")
    return [int(value) for value in raw]


def _trace_integer(trace: dict[str, object], key: str) -> int:
    value = trace.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Decision trace field {key!r} must be an integer.")
    return int(value)


def _signed_slippage_bps(
    execution_price: float | None, benchmark_price: float | None, side_multiplier: int
) -> float | None:
    if execution_price is None or benchmark_price is None or benchmark_price <= 0:
        return None
    return 10_000.0 * side_multiplier * (execution_price - benchmark_price) / benchmark_price


def _side_multiplier(side: str) -> int:
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    raise ValueError("Side must be 'buy' or 'sell'.")
