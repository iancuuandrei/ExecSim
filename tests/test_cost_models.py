from __future__ import annotations

import pytest

from execsim.costs import CostParameter, LinearTemporaryImpactModel


def test_zero_cost_recovers_reference_and_buy_sell_costs_are_symmetric() -> None:
    zero = LinearTemporaryImpactModel()
    buy = zero.quote(side="buy", reference_price=100.0, executed_qty=100, market_volume=1_000)
    sell = zero.quote(side="sell", reference_price=100.0, executed_qty=100, market_volume=1_000)

    assert buy.execution_price == sell.execution_price == 100.0
    assert buy.total_modeled_cost == sell.total_modeled_cost == 0.0


def test_linear_temporary_impact_formula_and_direction() -> None:
    model = LinearTemporaryImpactModel(
        half_spread=CostParameter(0.02), temporary_impact=CostParameter(0.50)
    )
    buy = model.quote(side="buy", reference_price=100.0, executed_qty=100, market_volume=1_000)
    sell = model.quote(side="sell", reference_price=100.0, executed_qty=100, market_volume=1_000)

    assert buy.temporary_impact_per_share == pytest.approx(0.05)
    assert buy.execution_price == pytest.approx(100.07)
    assert sell.execution_price == pytest.approx(99.93)
    assert buy.spread_cost == sell.spread_cost == pytest.approx(2.0)
    assert buy.temporary_impact_cost == sell.temporary_impact_cost == pytest.approx(5.0)


def test_impact_cost_is_monotone_in_quantity_and_decreases_with_volume() -> None:
    model = LinearTemporaryImpactModel(temporary_impact=CostParameter(1.0))
    small = model.quote(side="buy", reference_price=10.0, executed_qty=10, market_volume=100)
    large = model.quote(side="buy", reference_price=10.0, executed_qty=20, market_volume=100)
    liquid = model.quote(side="buy", reference_price=10.0, executed_qty=20, market_volume=200)

    assert large.temporary_impact_cost > small.temporary_impact_cost
    assert liquid.temporary_impact_cost < large.temporary_impact_cost


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_cost_parameters_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        CostParameter(value)
