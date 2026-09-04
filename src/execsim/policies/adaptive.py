from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from execsim.optimization import OptimalExecutionProblem, OptimalExecutionWorkspace
from execsim.policies.models import DecisionContext, PolicyDecision


@dataclass(slots=True)
class AdaptiveMPCPolicy:
    risk_aversion: float = 0.0
    tracking_penalty: float = 0.0
    half_spread: float = 0.0
    temporary_impact: float = 0.01
    volatility: float = 0.01
    policy_name: str = "mpc"
    _warm_start: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _workspace: OptimalExecutionWorkspace | None = field(default=None, init=False, repr=False)
    _decision_number: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._warm_start = None
        self._decision_number = 0

    def decide(self, context: DecisionContext) -> PolicyDecision:
        if context.forecast is None:
            raise ValueError("Adaptive MPC requires a point-in-time volume forecast.")
        n = context.remaining_buckets
        if self._workspace is None or self._workspace.maximum_horizon != n + self._decision_number:
            self._workspace = OptimalExecutionWorkspace(n, validation_level="structural")
        problem = OptimalExecutionProblem(
            quantity=max(context.remaining_inventory, 1),
            forecast_volumes=np.asarray(context.forecast.expected_volumes, dtype=float),
            forecast_volatilities=np.full(n, self.volatility),
            half_spreads=np.full(n, self.half_spread),
            temporary_impacts=np.full(n, self.temporary_impact),
            max_participation_rate=context.constraints.planned_participation_rate,
            risk_aversion=self.risk_aversion,
            tracking_penalty=self.tracking_penalty,
            forecast_weights=np.asarray(context.forecast.normalized_shares, dtype=float),
        )
        warm = (
            self._warm_start
            if self._warm_start is not None and len(self._warm_start) == n
            else None
        )
        result = self._workspace.solve(problem, warm_start=warm)
        self._warm_start = result.continuous_quantities[1:].copy()
        self._decision_number += 1
        action = min(int(result.integer_quantities[0]), context.remaining_inventory)
        expected_volume = float(context.forecast.expected_volumes[0])
        inventory_after = context.remaining_inventory - action
        capacity = int(result.capacities.sum())
        required_participation = (
            context.remaining_inventory / context.forecast.expected_remaining_volume
            if context.forecast.expected_remaining_volume > 0
            else float("inf")
        )
        decision_id = f"mpc-{self._decision_number:04d}"
        return PolicyDecision(
            policy_name=self.policy_name,
            planned_quantity=action,
            forecast_id=context.forecast.forecaster_id,
            decision_id=decision_id,
            trace={
                "decision_id": decision_id,
                "decision_timestamp": context.current_timestamp,
                "remaining_inventory": context.remaining_inventory,
                "forecast_remaining_volume": context.forecast.expected_remaining_volume,
                "forecast_capacity": capacity,
                "required_average_future_participation": required_participation,
                "feasibility_status": (
                    "feasible" if result.predicted_capacity_shortfall == 0 else "capacity_shortfall"
                ),
                "predicted_capacity_shortfall": result.predicted_capacity_shortfall,
                "risk_aversion": self.risk_aversion,
                "tracking_penalty": self.tracking_penalty,
                "planned_first_action": action,
                "expected_spread_cost": self.half_spread * action,
                "expected_temporary_impact_cost": (
                    self.temporary_impact * action**2 / max(expected_volume, 1.0)
                ),
                "expected_inventory_risk": (
                    self.risk_aversion * self.volatility**2 * inventory_after**2
                ),
                "solver_status": result.diagnostics.status,
                "solver_iterations": result.diagnostics.iterations,
                "solver_primal_residual": result.diagnostics.primal_residual,
                "solver_dual_residual": result.diagnostics.dual_residual,
                "solver_absolute_tolerance": result.diagnostics.absolute_tolerance,
                "solver_relative_tolerance": result.diagnostics.relative_tolerance,
                "solve_time_seconds": result.diagnostics.solve_time_seconds,
                "matrix_construction_time_seconds": (
                    result.diagnostics.matrix_construction_time_seconds
                ),
                "solver_setup_time_seconds": result.diagnostics.solver_setup_time_seconds,
                "solver_update_time_seconds": result.diagnostics.solver_update_time_seconds,
                "eigenvalue_validation_time_seconds": (
                    result.diagnostics.eigenvalue_validation_time_seconds
                ),
                "integer_projection_time_seconds": (
                    result.diagnostics.integer_projection_time_seconds
                ),
                "solver_workspace_reused": result.diagnostics.workspace_reused,
                "solver_validation_level": result.diagnostics.validation_level,
                "forecast_id": context.forecast.forecaster_id,
                "forecast_training_cutoff": context.forecast.training_data_cutoff,
            },
        )
