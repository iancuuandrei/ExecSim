from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ParameterProvenance(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    ASSUMED = "assumed"
    EXTERNALLY_SUPPLIED = "externally_supplied"


@dataclass(frozen=True, slots=True)
class CostParameter:
    """A non-negative currency-per-share cost input and its provenance."""

    value: float
    provenance: ParameterProvenance = ParameterProvenance.ASSUMED
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("Cost parameter values must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class CostQuote:
    reference_price: float
    half_spread: float
    temporary_impact_per_share: float
    execution_price: float
    spread_cost: float
    temporary_impact_cost: float
    total_modeled_cost: float


@runtime_checkable
class ExecutionCostModel(Protocol):
    """Public contract for a causal, per-bucket execution-cost model."""

    @property
    def model_id(self) -> str: ...

    def quote(
        self,
        *,
        side: str,
        reference_price: float,
        executed_qty: int,
        market_volume: float,
    ) -> CostQuote: ...
