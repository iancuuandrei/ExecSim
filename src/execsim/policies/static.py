from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execsim.optimization import (
    OptimalExecutionProblem,
    almgren_chriss_continuous_schedule,
    project_to_integer_capacities,
    solve_optimal_execution,
)
from execsim.orders import ParentOrder
from execsim.policies.models import DecisionContext, PolicyDecision, SchedulePlan


def _reconcile_weights(weights: np.ndarray, quantity: int) -> np.ndarray:
    positive = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if positive.ndim != 1 or not positive.size or positive.sum() <= 0:
        raise ValueError("Schedule weights must contain positive mass.")
    continuous = quantity * positive / positive.sum()
    capacities = np.full(len(positive), quantity, dtype=np.int64)
    return project_to_integer_capacities(continuous, capacities, quantity)


@dataclass(frozen=True, slots=True)
class TwapPolicy:
    policy_name: str = "twap"

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        quantities = _reconcile_weights(
            np.ones(context.remaining_buckets), context.remaining_inventory
        )
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=context.future_timestamps,
            quantities=tuple(int(value) for value in quantities),
            feasible_planned_quantity=context.remaining_inventory,
        )


@dataclass(frozen=True, slots=True)
class HistoricalVwapPolicy:
    policy_name: str = "vwap"

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        del parent_order
        if context.forecast is None:
            raise ValueError("Historical VWAP requires a point-in-time volume forecast.")
        quantities = _reconcile_weights(
            np.asarray(context.forecast.normalized_shares), context.remaining_inventory
        )
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=context.future_timestamps,
            quantities=tuple(int(value) for value in quantities),
            feasible_planned_quantity=context.remaining_inventory,
            forecast_id=context.forecast.forecaster_id,
            warnings=context.forecast.warnings,
        )


@dataclass(frozen=True, slots=True)
class PovPolicy:
    target_participation_rate: float
    policy_name: str = "pov"

    def __post_init__(self) -> None:
        if not 0 <= self.target_participation_rate <= 1:
            raise ValueError("target_participation_rate must be in [0, 1].")

    def decide(
        self, context: DecisionContext, *, observable_current_volume: float
    ) -> PolicyDecision:
        requested = int(np.floor(self.target_participation_rate * observable_current_volume))
        return PolicyDecision(
            policy_name=self.policy_name,
            planned_quantity=min(requested, context.remaining_inventory),
            trace={
                "causality": "participates_as_current_bucket_volume_materializes",
                "target_participation_rate": self.target_participation_rate,
            },
        )


@dataclass(frozen=True, slots=True)
class AlmgrenChrissPolicy:
    temporary_quadratic_coefficient: float
    volatility: float
    risk_aversion: float
    policy_name: str = "almgren-chriss"

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        del parent_order
        continuous = almgren_chriss_continuous_schedule(
            context.remaining_inventory,
            context.remaining_buckets,
            temporary_quadratic_coefficient=self.temporary_quadratic_coefficient,
            volatility=self.volatility,
            risk_aversion=self.risk_aversion,
        )
        capacities = np.full(context.remaining_buckets, context.remaining_inventory, dtype=np.int64)
        quantities = project_to_integer_capacities(
            continuous, capacities, context.remaining_inventory, tolerance=1e-5
        )
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=context.future_timestamps,
            quantities=tuple(int(value) for value in quantities),
            feasible_planned_quantity=context.remaining_inventory,
        )


@dataclass(frozen=True, slots=True)
class ConstrainedOptimalPolicy:
    risk_aversion: float = 0.0
    tracking_penalty: float = 0.0
    half_spread: float = 0.0
    temporary_impact: float = 0.01
    volatility: float = 0.01
    policy_name: str = "optimal"

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        del parent_order
        if context.forecast is None:
            raise ValueError("Constrained optimal policy requires a point-in-time forecast.")
        n = context.remaining_buckets
        problem = OptimalExecutionProblem(
            quantity=context.remaining_inventory,
            forecast_volumes=np.asarray(context.forecast.expected_volumes, dtype=float),
            forecast_volatilities=np.full(n, self.volatility),
            half_spreads=np.full(n, self.half_spread),
            temporary_impacts=np.full(n, self.temporary_impact),
            max_participation_rate=context.constraints.planned_participation_rate,
            risk_aversion=self.risk_aversion,
            tracking_penalty=self.tracking_penalty,
            forecast_weights=np.asarray(context.forecast.normalized_shares, dtype=float),
        )
        result = solve_optimal_execution(problem)
        warnings = (
            (f"predicted_capacity_shortfall={result.predicted_capacity_shortfall}",)
            if result.predicted_capacity_shortfall
            else ()
        )
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=context.future_timestamps,
            quantities=tuple(int(value) for value in result.integer_quantities),
            feasible_planned_quantity=result.feasible_quantity,
            predicted_capacity_shortfall=result.predicted_capacity_shortfall,
            forecast_id=context.forecast.forecaster_id,
            solver_diagnostics=result.diagnostics,
            warnings=warnings,
        )
