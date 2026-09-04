from __future__ import annotations

import math
from dataclasses import dataclass, field

from execsim.costs.models import CostParameter, CostQuote


@dataclass(frozen=True, slots=True)
class LinearTemporaryImpactModel:
    """Half-spread plus convex linear-in-participation temporary impact.

    ``temporary_impact`` is currency/share at 100% bucket participation.
    """

    half_spread: CostParameter = field(default_factory=lambda: CostParameter(0.0))
    temporary_impact: CostParameter = field(default_factory=lambda: CostParameter(0.0))
    epsilon_volume: float = 1.0
    model_id: str = "linear-temporary-impact-v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon_volume) or self.epsilon_volume <= 0:
            raise ValueError("epsilon_volume must be finite and positive.")

    def quote(
        self,
        *,
        side: str,
        reference_price: float,
        executed_qty: int,
        market_volume: float,
    ) -> CostQuote:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError("reference_price must be finite and positive.")
        if isinstance(executed_qty, bool) or not isinstance(executed_qty, int):
            raise TypeError("executed_qty must be an integer.")
        if executed_qty < 0:
            raise ValueError("executed_qty must be non-negative.")
        if not math.isfinite(market_volume) or market_volume < 0:
            raise ValueError("market_volume must be finite and non-negative.")

        denominator = max(float(market_volume), self.epsilon_volume)
        impact_per_share = self.temporary_impact.value * executed_qty / denominator
        direction = 1.0 if side == "buy" else -1.0
        execution_price = reference_price + direction * (self.half_spread.value + impact_per_share)
        spread_cost = self.half_spread.value * executed_qty
        impact_cost = impact_per_share * executed_qty
        return CostQuote(
            reference_price=reference_price,
            half_spread=self.half_spread.value,
            temporary_impact_per_share=impact_per_share,
            execution_price=execution_price,
            spread_cost=spread_cost,
            temporary_impact_cost=impact_cost,
            total_modeled_cost=spread_cost + impact_cost,
        )
