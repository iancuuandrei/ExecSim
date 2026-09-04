from __future__ import annotations

from dataclasses import dataclass

from execsim.orders import ParentOrder
from execsim.policies.models import DecisionContext, SchedulePlan
from execsim.policies.static import HistoricalVwapPolicy


@dataclass(frozen=True, slots=True)
class OracleVwapPolicy:
    """Evaluation-only hindsight volume schedule; never a deployable policy."""

    policy_name: str = "oracle-vwap-evaluation-only"
    evaluation_only: bool = True

    def create_plan(self, parent_order: ParentOrder, context: DecisionContext) -> SchedulePlan:
        if context.forecast is None or not context.forecast.forecaster_id.startswith("oracle-"):
            raise ValueError("OracleVwapPolicy requires an explicitly labeled oracle forecast.")
        plan = HistoricalVwapPolicy().create_plan(parent_order, context)
        return SchedulePlan(
            policy_name=self.policy_name,
            timestamps=plan.timestamps,
            quantities=plan.quantities,
            feasible_planned_quantity=plan.feasible_planned_quantity,
            forecast_id=context.forecast.forecaster_id,
            warnings=context.forecast.warnings,
        )


DEPLOYABLE_POLICY_NAMES = (
    "twap",
    "vwap",
    "pov",
    "almgren-chriss",
    "optimal",
    "mpc",
)
