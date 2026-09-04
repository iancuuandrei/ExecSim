from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from execsim.forecasting.models import VolumeForecast
from execsim.orders import ParentOrder


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    planned_participation_rate: float = 1.0
    hard_participation_rate: float = 1.0
    timezone: str = "America/New_York"

    def __post_init__(self) -> None:
        for name, value in (
            ("planned_participation_rate", self.planned_participation_rate),
            ("hard_participation_rate", self.hard_participation_rate),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")


@dataclass(frozen=True, slots=True)
class DecisionContext:
    current_timestamp: pd.Timestamp
    decision_timing: str
    remaining_inventory: int
    elapsed_buckets: int
    remaining_buckets: int
    observations: pd.DataFrame
    future_timestamps: tuple[pd.Timestamp, ...]
    forecast: VolumeForecast | None
    constraints: ExecutionConstraints
    config_version: str = "v1"
    model_version: str = "deterministic-v1"

    def __post_init__(self) -> None:
        if self.current_timestamp.tzinfo is None:
            raise ValueError("Decision timestamps must be timezone-aware.")
        if self.remaining_inventory < 0 or self.elapsed_buckets < 0 or self.remaining_buckets <= 0:
            raise ValueError("Decision inventory and bucket counts are invalid.")
        if len(self.future_timestamps) != self.remaining_buckets:
            raise ValueError("future_timestamps must match remaining_buckets.")
        if self.future_timestamps[0] != self.current_timestamp:
            raise ValueError("The first future bucket must be the current decision timestamp.")
        if "timestamp" in self.observations.columns and not self.observations.empty:
            observed = pd.to_datetime(self.observations["timestamp"])
            if observed.dt.tz is None or (observed >= self.current_timestamp).any():
                raise ValueError(
                    "Decision observations must be timezone-aware and strictly in the past."
                )
        if self.forecast is not None:
            if self.forecast.generated_at > self.current_timestamp:
                raise ValueError("A decision cannot use a forecast generated in the future.")
            if self.forecast.bucket_timestamps != self.future_timestamps:
                raise ValueError("Forecast and decision horizons must match exactly.")


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    policy_name: str
    timestamps: tuple[pd.Timestamp, ...]
    quantities: tuple[int, ...]
    feasible_planned_quantity: int
    predicted_capacity_shortfall: int = 0
    forecast_id: str | None = None
    solver_diagnostics: object | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.quantities) or not self.timestamps:
            raise ValueError("Plan timestamps and quantities must be equal and non-empty.")
        if any(quantity < 0 for quantity in self.quantities):
            raise ValueError("Planned quantities must be non-negative.")
        if sum(self.quantities) != self.feasible_planned_quantity:
            raise ValueError("Planned quantities must reconcile to feasible_planned_quantity.")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_name: str
    planned_quantity: int
    forecast_id: str | None = None
    decision_id: str | None = None
    trace: dict[str, object] | None = None


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Contract for a policy that creates one causal schedule for the horizon."""

    policy_name: str

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan: ...


@runtime_checkable
class AdaptiveExecutionPolicy(Protocol):
    """Contract for a policy that chooses one action at every decision point."""

    policy_name: str

    def reset(self) -> None: ...

    def decide(self, context: DecisionContext) -> PolicyDecision: ...
