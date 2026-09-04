from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from execsim.costs import CostParameter, LinearTemporaryImpactModel
from execsim.forecasting import HistoricalProfileForecaster
from execsim.orders import ParentOrder
from execsim.policies import (
    AdaptiveMPCPolicy,
    ConstrainedOptimalPolicy,
    ExecutionConstraints,
    HistoricalVwapPolicy,
    PovPolicy,
)
from execsim.simulator import simulate_policy


def _bars(session: str, volumes: list[int], prices: list[float] | None = None) -> pd.DataFrame:
    prices = prices or [100.0 + index for index in range(len(volumes))]
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(volumes),
            "timestamp": pd.date_range(
                f"{session} 09:30", periods=len(volumes), freq="min", tz="America/New_York"
            ),
            "open": prices,
            "high": [value + 0.1 for value in prices],
            "low": [value - 0.1 for value in prices],
            "close": prices,
            "volume": volumes,
            "trade_count": [10] * len(volumes),
            "vwap": prices,
        }
    )


def _order(quantity: int = 12) -> ParentOrder:
    return ParentOrder("AAPL", "buy", quantity, date(2026, 3, 16), time(9, 30), time(9, 34))


def _forecaster() -> HistoricalProfileForecaster:
    history = pd.concat(
        [_bars("2026-03-12", [100, 100, 100, 100]), _bars("2026-03-13", [100] * 4)],
        ignore_index=True,
    )
    return HistoricalProfileForecaster(history)


def test_vwap_optimal_and_mpc_are_interchangeable_in_one_simulator() -> None:
    bars = _bars("2026-03-16", [100] * 4)
    constraints = ExecutionConstraints(0.2, 0.2)
    policies = [
        HistoricalVwapPolicy(),
        ConstrainedOptimalPolicy(temporary_impact=0.1),
        AdaptiveMPCPolicy(temporary_impact=0.1),
    ]

    results = [
        simulate_policy(
            parent_order=_order(),
            bars=bars,
            policy=policy,
            constraints=constraints,
            forecast_provider=_forecaster(),
        )
        for policy in policies
    ]

    assert [result.summary.strategy for result in results] == ["vwap", "optimal", "mpc"]
    assert all(result.summary.filled_qty == 12 for result in results)
    assert results[1].summary.n_optimization_decisions == 1
    assert results[2].summary.n_optimization_decisions == 4
    assert len(results[2].decision_trace) == 4
    assert (results[2].decision_trace["forecast_training_cutoff"] < date(2026, 3, 16)).all()


def test_pov_uses_current_volume_and_respects_hard_cap_and_inventory() -> None:
    result = simulate_policy(
        parent_order=_order(quantity=20),
        bars=_bars("2026-03-16", [10, 20, 0, 100]),
        policy=PovPolicy(0.5),
        constraints=ExecutionConstraints(0.5, 0.25),
    )

    assert result.execution_log["planned_qty"].tolist() == [5, 10, 0, 13]
    assert result.execution_log["executed_qty"].tolist() == [2, 5, 0, 13]
    assert result.summary.unfilled_qty == 0
    assert result.summary.maximum_participation <= 0.25


def test_cost_decomposition_reconciles_for_buys_and_sells() -> None:
    model = LinearTemporaryImpactModel(
        half_spread=CostParameter(0.01), temporary_impact=CostParameter(0.5)
    )
    bars = _bars("2026-03-16", [100] * 4, [100.0, 101.0, 102.0, 103.0])
    for side in ("buy", "sell"):
        order = ParentOrder("AAPL", side, 12, date(2026, 3, 16), time(9, 30), time(9, 34))
        result = simulate_policy(
            parent_order=order,
            bars=bars,
            policy=HistoricalVwapPolicy(),
            constraints=ExecutionConstraints(1.0, 1.0),
            cost_model=model,
            forecast_provider=_forecaster(),
        )
        summary = result.summary
        assert summary.cost_reconciliation_residual == pytest.approx(0.0, abs=1e-9)
        assert summary.modeled_spread_cost > 0
        assert summary.modeled_temporary_impact_cost > 0


@pytest.mark.parametrize("fault", ["duplicate", "naive", "negative_volume", "nan_price"])
def test_simulator_rejects_malformed_bars(fault: str) -> None:
    bars = _bars("2026-03-16", [100] * 4)
    if fault == "duplicate":
        bars.loc[1, "timestamp"] = bars.loc[0, "timestamp"]
    elif fault == "naive":
        bars["timestamp"] = bars["timestamp"].dt.tz_localize(None)
    elif fault == "negative_volume":
        bars.loc[1, "volume"] = -1
    else:
        bars.loc[1, "open"] = float("nan")

    with pytest.raises(ValueError):
        simulate_policy(parent_order=_order(), bars=bars, policy=PovPolicy(0.1))
